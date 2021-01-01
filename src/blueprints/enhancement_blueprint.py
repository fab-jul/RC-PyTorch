"""
Copyright 2020, ETH Zurich

This file is part of RC-PyTorch.

RC-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

RC-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RC-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
from typing import Optional

from helpers import tau_optim
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as transforms

import pytorch_ext as pe
import vis.summarizable_module
from blueprints.classifier_blueprint import load_classifier
from criterion.logistic_mixture import DiscretizedMixLogisticLoss
from dataloaders import images_loader
from helpers import cin_bins
from helpers.global_config import global_config
from helpers.pad import pad
from helpers.quantized_tensor import SymbolTensor, NormalizedTensor
from modules import prob_clf, conditional_instance_norm
from modules_enh.enhancement_network import EnhancementNetwork
from enum import Enum


CIN_BIN_MIN = 0.5
CIN_BIN_MAX = 3.5

_KILL_BORDER = int(os.environ.get('KILL_PAD', 0))

InputTensors = namedtuple('InputTensors', ['x_n',
                                           'bpps',
                                           'n_sp_pre_pad'])  # number of subpixels before padding

EnhancementLoss = namedtuple(
        'EnhancementLoss',
        ['total_loss',
         'bpsp_base',
         'bpsp_residual'])


class QStrategy(Enum):
    MIN = 'MIN'
    CLF = 'CLF'
    CLF_ONLY = 'CLF_ONLY'
    FIXED = 'FIXED'


def enhancement_loss_lt(a: EnhancementLoss, b: EnhancementLoss):
    return (a.bpsp_base + a.bpsp_residual) < (b.bpsp_base + b.bpsp_residual)


_ZERO = torch.tensor(0.)


class EnhancementOut(object):
    def __init__(self, network_out: prob_clf.NetworkOutput, x_r: NormalizedTensor, x_l: NormalizedTensor):
        self.network_out = network_out
        self.x_r = x_r.to_sym()
        res = self.x_r.t - x_l.to_sym().t
        self.res_sym = SymbolTensor(res, L=511, centered=True)
        self.res = self.res_sym.to_norm()
        self._mean_img = None

    def get_mean_img(self, loss: DiscretizedMixLogisticLoss):
        if self._mean_img is None:
            self._mean_img = extract_mean_image_corrected(
                    self.res, self.network_out, loss)
        return self._mean_img


def extract_mean_image_corrected(res: NormalizedTensor, l: prob_clf.NetworkOutput, loss: DiscretizedMixLogisticLoss):
    _, logit_probs, corrected_means, _ = loss._extract_non_shared(res, l)
    if l.means.shape[2] > 1:  # K > 1 -> must average with pis
        logit_pis_sm = F.softmax(logit_probs, dim=2)  # NCKHW, pi_k
        means_scaled = corrected_means * logit_pis_sm
        means = means_scaled.sum(2)
    else:
        means = corrected_means[:, :, 0, ...]
    return means.clamp(-1, 1)


class EnhancementLosses(vis.summarizable_module.SummarizableModule):
    def __init__(self, config_en, is_testing):
        super(EnhancementLosses, self).__init__()
        self.config_en = config_en
        self.is_testing = is_testing

        self.loss_dmol_rgb = DiscretizedMixLogisticLoss(rgb_scale=True, L=511)

        self.tau_optimization_helper: Optional[tau_optim.TauOptimizationHelper] = None

        self.gains = []

    def print_tau_optimization_summary(self):
        if self.tau_optimization_helper:
            self.tau_optimization_helper.print_summary()


    def set_eval(self):
        pass

    def forward(self, out: EnhancementOut, num_subpixels_before_pad=None, base_bpp=0.) -> EnhancementLoss:
        num_subpixels = int(np.prod(out.res.get().shape))

        if num_subpixels_before_pad:
            assert num_subpixels_before_pad <= num_subpixels, num_subpixels_before_pad
            num_subpixels = num_subpixels_before_pad

        base_bpsp = base_bpp / 3

        conversion = np.log(2.) * num_subpixels

        # with cuda_timer.execute('>>> nll'):
        if self.tau_optimization_helper:
            nll, tau_overhead_bytes = self.tau_optimization_helper.optimize(out.res, out.network_out)
        else:
            nll = self.loss_dmol_rgb(out.res, out.network_out, scale=0)
            tau_overhead_bytes = None

        # We don't take the mean because of the num_subpixels_before_pad
        bpsp_residual = nll.sum() / conversion
        if tau_overhead_bytes is not None:
            tau_overhead_bpsp = tau_overhead_bytes * 8 / num_subpixels
            bpsp_residual += tau_overhead_bpsp

        # Note: only enabled as heavy log
        scalars = {'costs/bpsp_base': base_bpsp}
        self.summarizer.register_scalars('train', scalars)

        self.summarizer.register_images('auto', {'res_img': lambda: out.res.get()}, normalize=True, only_once=True)
        return EnhancementLoss(total_loss=bpsp_residual,
                               bpsp_base=base_bpsp,
                               bpsp_residual=bpsp_residual)


class EnhancementBlueprint(vis.summarizable_module.SummarizableModule):
    def __init__(self, config_en, is_testing=False):
        super(EnhancementBlueprint, self).__init__()

        self.net = EnhancementNetwork(config_en)
        self.net = self.net.to(pe.DEVICE)
        self.config_en = config_en

        self.clf = None
        self.qstrategy = QStrategy.MIN

        self.losses = EnhancementLosses(config_en, is_testing)

        global_config.assert_only_one('cinorm', 'cin_eb', 'cgdn')

        self.cin_style = None
        if global_config.get('cinorm', False):
            self.cin_style = 'cinorm'
        elif global_config.get('cin_eb', False):
            self.cin_style = 'evenbins'
            self.cin_q = cin_bins.Quantizer(global_config['cin_eb'])
        elif global_config.get('cgdn', False):
            self.cin_style = 'cgdn'
            self.cin_q = cin_bins.Quantizer(global_config['cgdn'])
        print('EB: self.cin_style =', self.cin_style)

        self.padding_fac = self.get_padding_fac()
        print('***' * 10)
        print('*** Padding by a factor', self.padding_fac)
        print('***' * 10)

    def enable_tau_optimization(self):
        self.losses.tau_optimization_helper = tau_optim.TauOptimizationHelper(
            self.losses.loss_dmol_rgb)

    def add_classifier(self, clf_ckpt_p):
        if self.clf is None:
            self.clf = load_classifier(clf_ckpt_p)

    @staticmethod
    def read_evenly_spaced_bins(config_dl):
        flag = global_config.get('cin_eb', None) or global_config.get('cgdn', None)
        if flag and flag.startswith('auto'):
            flag_name = 'cin_eb' if global_config.get('cin_eb', None) else 'cgdn'
            nb = int(flag.replace('auto', ''))
            # creates if needed
            pkl_p = cin_bins.make_bin_pkl(config_dl.imgs_dir_train['compressed'], nb)
            print(f'Setting {flag_name} = {pkl_p}')
            global_config[flag_name] = pkl_p

    def set_eval(self):
        self.net.eval()
        self.losses.set_eval()

    # TODO:
    # for side_information, the encoder has to save it somewhere!
    def forward_lossy(self, x_l, bpps):
        # pad for RC
        x_l.t, undo_pad = self.pad_undo(x_l.t)
        with conditional_instance_norm.cin_classes_context(self.cin_style is not None, self._get_cin_classes, bpps):
            network_out: prob_clf.NetworkOutput = self.net(x_l)
            # undo pad
            return prob_clf.NetworkOutput(*map(undo_pad, network_out))

    def pad(self, x):
        raw, _ = pad(x, self.padding_fac, mode='constant')
        return raw

    def pad_undo(self, x, mode='reflect'):
        raw, undo_pad = pad(x, self.padding_fac, mode=mode)
        return raw, undo_pad

    def forward(self, xs: tuple, bpps) -> EnhancementOut:
        """
        :param xs: tuple of NCHW NormalizedTensor of (raw, lossy)
        :param auto_recurse: int, how many times the last scales should be applied again. Used for RGB Shared.
        :return: layers.multiscale.Out
        """
        with conditional_instance_norm.cin_classes_context(self.cin_style is not None, self._get_cin_classes, bpps):
            x_r, x_l = xs
            side_information = self.net.extract_side_information(x_r, x_l)
            network_out: prob_clf.NetworkOutput = self.net(x_l, side_information)
            return EnhancementOut(network_out, x_r, x_l)

    def _get_cin_classes(self, bpps):
        if self.cin_style is None:
            return None
        if self.cin_style == 'cinorm':
            return get_cin_classes(bpps / 3)
        if self.cin_style == 'evenbins':
            return self.cin_q.quantize_batch_one_hot(bpps / 3)
        if self.cin_style == 'cgdn':
            return self.cin_q.quantize_batch(bpps / 3)
        raise ValueError('invalid', self.cin_style)


    @staticmethod
    def get_test_dataset_transform(crop):
        # TODO(enh)
        raise NotImplementedError
        img_to_tensor_t = [images_loader.IndexImagesDataset.to_tensor_uint8_transform()]
        if global_config.get('ycbcr', False):
            print('Adding ->YCbCr to Testset')
            t = transforms.Lambda(lambda pil_img: pil_img.convert('YCbCr'))
            img_to_tensor_t.insert(0, t)
        if crop:
            print(f'Cropping Testset: {crop}')
            img_to_tensor_t.insert(0, transforms.CenterCrop(crop))
        return transforms.Compose(img_to_tensor_t)

    def sample_forward(self, network_out: prob_clf.NetworkOutput):
        return self.losses.loss_dmol_rgb.sample(network_out)

    @staticmethod
    def add_image_summaries(sw, out: EnhancementOut, global_step, prefix):
        pass  # TODO(enh) maybe
        # tag = lambda t: sw.pre(prefix, t)
        # is_train = prefix == 'train'
        # for scale, z_i in enumerate(out.z[1:], 1):  # start from 1, as 0 is RGB
        #     S_i: SymbolTensor = z_i.to_sym()
        #     sw.add_image(tag('bn/{}'.format(scale)), new_bottleneck_summary(S_i), global_step)
        #     # # This will only trigger for the final scale, where P_i is the uniform distribution.
        #     # # With this, we can check how accurate the uniform assumption is (hint: not very)
        #     # is_logits = (scale == (len(out.z) - 1))
        #     # if is_logits and is_train:
        #     #     P_i = out.get_uniform_P()
        #     #     with sw.add_figure_ctx(tag('histo_out/{}'.format(scale)), global_step) as plt:
        #     #         add_ps_summaries(S_i, get_p_y(P_i), plt)

    def get_padding_fac(self):
        return 2 if global_config.get('down_up', None) else 0

    def unpack_batch_pad(self, img_or_imgbatch) -> (NormalizedTensor, SymbolTensor):
        raw, compressed, bpps = self.unpack_batch_light(img_or_imgbatch)
        return self.pad_pack(raw, compressed, bpps)

    def pad_pack(self, raw, compressed, bpps) -> InputTensors:
        """ Pad iimages and pack into a InputTensors instance

        :param raw: Batch of raw input images
        :param compressed: Output of compressing images in `raw` with BPG.
        :param bpps: The bitrates of the images.
        :return: InputTensors
        """
        assert raw.shape == compressed.shape

        num_subpixels_before_pad = np.prod(raw.shape)

        if self.padding_fac:
            raw = self.pad(raw)
            compressed = self.pad(compressed)

        assert len(raw.shape) == 4
        assert len(bpps.shape) == 1, (bpps.shape, raw.shape)

        s_c = SymbolTensor(compressed.long(), L=256)
        x_c = s_c.to_norm()
        s_r = SymbolTensor(raw.long(), L=256)
        x_r = s_r.to_norm()
        return InputTensors((x_r, x_c), bpps, num_subpixels_before_pad)

    def unpack_batch_light(self, img_or_imgbatch):
        raw = img_or_imgbatch['raw'].to(pe.DEVICE, non_blocking=True)  # uint8 or int16
        compressed = img_or_imgbatch['compressed'].to(pe.DEVICE, non_blocking=True)  # uint8 or int16
        bpps = torch.tensor(img_or_imgbatch['bpp']).to(pe.DEVICE).view(-1)  # 1d tensor of floats
        if len(raw.shape) == 3:
            raw.unsqueeze_(0)
            compressed.unsqueeze_(0)
        return raw, compressed, bpps

    def unpack(self, img_batch, fixed_first=None) -> InputTensors:
        raw = img_batch['raw'].to(pe.DEVICE, non_blocking=True)  # uint8 or int16
        compressed = img_batch['compressed'].to(pe.DEVICE, non_blocking=True)  # uint8 or int16
        bpps = img_batch['bpp'].to(pe.DEVICE)  # 1d tensor of floats

        if fixed_first is not None:
            raw[0, ...] = fixed_first['raw']
            compressed[0, ...] = fixed_first['compressed']
            bpps[0] = fixed_first['bpp']

        num_subpixels_before_pad = np.prod(raw.shape)

        if self.padding_fac:
            raw = self.pad(raw)
            compressed = self.pad(compressed)

        s_c = SymbolTensor(compressed.long(), L=256)
        x_c = s_c.to_norm()
        s_r = SymbolTensor(raw.long(), L=256)
        x_r = s_r.to_norm()

        return InputTensors((x_r, x_c), bpps, num_subpixels_before_pad)


# def new_bottleneck_summary(s: SymbolTensor):
#     """
#     Grayscale bottleneck representation: Expects the actual bottleneck symbols.
#     :param s: NCHW
#     :return: [0, 1] image
#     """
#     s_raw, L = s.get(), s.L
#     assert s_raw.dim() == 4, s_raw.shape
#     s_raw = s_raw.detach().float().div(L)
#     grid = vis.grid.prep_for_grid(s_raw, channelwise=True)
#     assert len(grid) == s_raw.shape[1], (len(grid), s_raw.shape)
#     assert [g.max() <= 1 for g in grid], [g.max() for g in grid]
#     assert grid[0].dtype == torch.float32, grid.dtype
#     return torchvision.utils.make_grid(grid, nrow=5)
#
#
# def _assert_contains_symbol_indices(t, L):
#     """ assert 0 <= t < L """
#     assert 0 <= t.min() and t.max() < L, (t.min(), t.max())
#
#
# def add_ps_summaries(s: SymbolTensor, p_y, plt):
#     histo_s = pe.histogram(s.t, s.L)
#     p_x = histo_s / np.sum(histo_s)
#
#     assert p_x.shape == p_y.shape, (p_x.shape, p_y.shape)
#
#     histogram_plotter.plot_histogram([
#         ('p_x', p_x),
#         ('p_y', p_y),
#     ], plt)
#
#
# def get_p_y(y):
#     """
#     :param y: NLCHW float, logits
#     :return: L dimensional vector p
#     """
#     Ldim = 1
#     L = y.shape[Ldim]
#     y = y.detach()
#     p = F.softmax(y, dim=Ldim)
#     p = p.transpose(Ldim, -1)
#     p = p.contiguous().view(-1, L)  # nL
#     p = torch.mean(p, dim=0)  # L
#     return pe.tensor_to_np(p)

# B +- bin_width/2 lands in the same bin
# so, last bin i
def get_cin_classes(bpsps):
    """ bpps: (N,) tensor"""
    nB = global_config['cinorm']
    bpsps_q = _quantize(bpsps, CIN_BIN_MIN, CIN_BIN_MAX, nB)
    return pe.one_hot(bpsps_q, L=nB, Ldim=-1)


def _quantize(x, bound_low, bound_high, num_bins):
    bound_range = bound_high - bound_low
    return x.clamp(bound_low, bound_high).sub(bound_low).div(bound_range).mul(num_bins-1).round().long()


def test_quantize():
    import torch
    ls = np.linspace(CIN_BIN_MIN, CIN_BIN_MAX, num=20)
    x = torch.tensor([0.4, 0.49, 0.5, 0.51, 0.8, 3.7, 4, 4.7, 5, 4.9479])
    print()
    print(x)
    print(ls)
    print(_quantize(x, CIN_BIN_MIN, CIN_BIN_MAX, num_bins=20))
    a = torch.from_numpy(np.linspace(CIN_BIN_MIN, CIN_BIN_MAX, num=39)) + 0.1
    print(a)
    print(_quantize(a, CIN_BIN_MIN, CIN_BIN_MAX, num_bins=20))
