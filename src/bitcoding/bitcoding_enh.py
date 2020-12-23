"""
Copyright 2020, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
from collections import namedtuple

import numpy as np
import torch
from PIL import Image
from fjcommon import functools_ext as ft

import bitcoding.coders_helpers
import pytorch_ext as pe
from bitcoding.coders import ArithmeticCoder
from blueprints.enhancement_blueprint import EnhancementBlueprint, EnhancementOut
from dataloaders import images_loader
from dataloaders.images_loader import IndexImagesDataset, to_tensor_not_normalized
from helpers.quantized_tensor import NormalizedTensor, SymbolTensor
from lossy.other_codecs import bpg_compress_to, decode_bpg_to_png, bpp_of_bpg_image
from modules import prob_clf
from test import cuda_timer


# A random sequence of bytes used to separate the bitstreams of different scales
_MAGIC_VALUE_SEP = b'\x46\xE2\x84\x92'


EncodeOut = namedtuple('EncodeOut', 'img,actual_bpsp,info_str')


class Bitcoding(object):
    """
    Class to encode an image to a file and decode it. Saves timings of individual steps to `times`.
    If `compare_with_theory = True`, also compares actual bitstream size to size predicted by cross entropy. Note
    that this is slower because we need to evaluate the loss.

    A note about padding:
    Our framework is
        x -> BPG -> x_l -> RC -> P(r)
    because RC may downsample (depending on arch), we pad x_l at the input of RC. The output we then crop again.
    Padding happens in enhancement_blueprint.forward_lossy
    """
    def __init__(self, blueprint: EnhancementBlueprint, times: cuda_timer.StackTimeLogger, compare_with_theory=False):
        self.blueprint = blueprint
        self.compare_with_theory = compare_with_theory
        self.times = times
        self.K = self.blueprint.config_en.prob.K

        self.pil_to_1CHW_long = lambda pil: to_tensor_not_normalized(pil).unsqueeze(0).to(pe.DEVICE).long()

    def encode(self, img, pout) -> EncodeOut:
        """
        Encode image to disk at path `p`.
        :return:  tuple EncodeOut , where img is int64 1CHW
        """
        with torch.no_grad():
            with self.times.prefix_scope('E'):
                return self._encode(img, pout)

    @staticmethod
    def _path_for_bpg(p):
        return p + '_bpg'

    def _encode_bpg(self, pin, pout, q):
        print(f'Encoding {pin} with Q={q}')
        bpg_compress_to(pin, pout, q)
        # bpp = bpp_of_bpg_image(pout)
        actual_bpp = os.path.getsize(pout) * 8 / np.prod(Image.open(pin).size)
        return actual_bpp

    def _decode_bpg(self, p) -> NormalizedTensor:
        pout_png = p + '_topng.ppm'  # ppm Should be faster
        decode_bpg_to_png(p, pout_png)
        img = self.pil_to_1CHW_long(Image.open(pout_png))  # int64, 1CHW
        os.remove(pout_png)
        return SymbolTensor(img, L=256).to_norm()

    def _encode(self, pin, pout) -> EncodeOut:
        """
        :param pin:
        :param pout:
        :return:  tuple (img, actual_bpsp), where img is int64 1CHW
        """
        assert not os.path.isfile(pout)
        img = self.pil_to_1CHW_long(Image.open(pin))  # int64 1CHW pe.DEVICE tensor
        assert len(img.shape) == 4 and img.shape[0] == 1 and img.shape[1] == 3, img.shape

        # gt
        x_r = SymbolTensor(img, L=256).to_norm()

        if self.blueprint.clf is not None:
            with self.times.run('Q-Classifier'):
                q = self.blueprint.clf.get_q(x_r.get())
        else:
            q = 12  # TODO

        with self.times.run(f'BPG'):
            # img = img.float()
            # Encode BPG
            pout_bpg = self._path_for_bpg(pout)
            bpp_bpg = self._encode_bpg(pin, pout_bpg, q)
            # 1. sym -> norm (for l)
            x_l: NormalizedTensor = self._decode_bpg(pout_bpg)

        with self.times.run('[-] encode forwardpass'):
            # 1. sym -> norm (for r)
            network_out: prob_clf.NetworkOutput = self.blueprint.forward_lossy(x_l, torch.tensor([bpp_bpg], device=pe.DEVICE))
            # in here:
            # 2. norm -> sym (for l and r)
            out = EnhancementOut(network_out, x_r, x_l)

        if self.compare_with_theory:
            with self.times.run('[-] get loss'):
                num_subpixels_before_pad = np.prod(img.shape)
                loss_out = self.blueprint.losses(out,
                                                 num_subpixels_before_pad=num_subpixels_before_pad,
                                                 base_bpp=bpp_bpg)

        entropy_coding_bytes = []  # bytes used by different scales

        dmll = self.blueprint.losses.loss_dmol_rgb

        with open(pout, 'wb') as fout:
            with self.times.prefix_scope(f'RGB'):
                entropy_coding_bytes.append(
                        self.encode_rgb(dmll, out, fout))
                fout.write(_MAGIC_VALUE_SEP)

        num_subpixels = np.prod(img.shape)
        actual_num_bytes = os.path.getsize(pout) + os.path.getsize(pout_bpg)
        actual_bpsp = actual_num_bytes * 8 / num_subpixels

        if self.compare_with_theory:
            # TODO
            raise NotImplementedError
            # assumed_bpsps = [b * 8 / num_subpixels for b in entropy_coding_bytes]
            # tostr = lambda l: ' | '.join(map('{:.3f}'.format, l)) + f' => {sum(l):.3f}'
            # overhead = (sum(assumed_bpsps) / sum(loss_out.nonrecursive_bpsps) - 1) * 100
            # return f'Bitrates:\n' \
            #     f'theory:  {tostr(loss_out.nonrecursive_bpsps)}\n' \
            #     f'assumed: {tostr(list(reversed(assumed_bpsps)))} [{overhead:.2f}%]\n' \
            #     f'actual:                                => {actual_bpsp:.3f} [{actual_num_bytes} bytes]'
        else:
            return EncodeOut(img, actual_bpsp, None)

    def decode_to(self, pin, pout):
        with torch.no_grad():
            t = self.decode(pin)  # 1CHW
            assert t.min() >= 0 and t.max() <= 255, (t.min(), t.max())
            a = t.squeeze(0).permute(1, 2, 0).to(torch.uint8).detach().cpu().numpy()
            i = Image.fromarray(a)
            i.save(pout)

    def decode(self, pin):
        """
        :param pin:  Path where image is stored
        :return: Decoded image, as 1CHW, long
        """
        with torch.no_grad():
            with self.times.prefix_scope('D'):
                return self._decode(pin)

    def _decode(self, pin):
        pin_bpg = self._path_for_bpg(pin)
        with self.times.run('BPG'):
            x_l: NormalizedTensor = self._decode_bpg(pin_bpg)
        with open(pin, 'rb') as fin:
            dmll = self.blueprint.losses.loss_dmol_rgb

            with self.times.prefix_scope(f'RGB'):
                with self.times.run('get_P'):
                    actual_bpp = os.path.getsize(pin_bpg) * 8 / np.prod(np.prod(x_l.t.shape)/3)
                    network_out: prob_clf.NetworkOutput = self.blueprint.forward_lossy(
                            x_l, torch.tensor([actual_bpp], device=pe.DEVICE))
                    # l, dec_out_prev = self.blueprint.net.get_P(
                    #         scale, bn_prev, dec_out_prev)
                # NCHW [-1, 1], residual
                res_decoded = self.decode_rgb(dmll, network_out, fin)
            assert fin.read(4) == _MAGIC_VALUE_SEP  # assert valid file
        assert res_decoded is not None  # assert decoding worked

        res_decoded_sym = NormalizedTensor(res_decoded, L=511, centered=True).to_sym()
        img = x_l.to_sym().t + res_decoded_sym.t
        return img  # 1CHW int64

    def encode_rgb(self, dmll, out: EnhancementOut, fout):
        """ Encode scale `scale`. """
        bn = out.res.t  # [-1, 1]
        residual_sym: SymbolTensor = out.res_sym
        x_range = residual_sym.t_range

        # TODO: could do per channel
        # these are used as indices into the CDF
        # 1CHW
        residual_sym_truncated_raw = residual_sym.t - x_range[0]  # [0, b - a]

        if self.blueprint.losses.tau_optim:
            with self.times.run('encode get_tau'):
                tau = self.blueprint.losses._find_optimal_tau(out.res, out.network_out)
                out.network_out = prob_clf.copy_network_output(out.network_out, sigmas=out.network_out.sigmas * tau)
                tau = tau.squeeze()
                assert tau.shape == (3, self.K)
                write_tau(tau, fout)

        # TODO RM
        assert residual_sym_truncated_raw.min() == 0, residual_sym_truncated_raw.min()

        overhead_bytes = write_range(x_range, fout)
        # shape used for all channels!
        overhead_bytes += write_shape(residual_sym.t.shape, fout)

        # TODO: why?  # write_num_bytes_encoded ??
        overhead_bytes += 2 * residual_sym.t.shape[1]

        r = ArithmeticCoder(L=x_range[1] - x_range[0] + 1)

        # We encode channel by channel, because that's what's needed for the RGB scale. For s > 0, this could be done
        # in parallel for all channels
        def encoder(c, C_cur):
            # TODO: why is it in long?
            residual_sym_truncated_raw_c = residual_sym_truncated_raw[:, c, ...].to(torch.int16)
            encoded = r.range_encode(residual_sym_truncated_raw_c, cdf=C_cur, time_logger=self.times)
            write_num_bytes_encoded(len(encoded), fout)
            fout.write(encoded)
            # yielding always bottleneck and extra_info
            return bn[:, c, ...], len(encoded)

        with self.times.prefix_scope('encode scale'):
            with self.times.run('total'):
                _, entropy_coding_bytes_per_c = \
                    self.code_with_cdf(out.network_out, bn.shape, encoder, dmll, x_range)

        return sum(entropy_coding_bytes_per_c)

    def decode_rgb(self, dmll, l: prob_clf.NetworkOutput, fin):

        if self.blueprint.losses.tau_optim:
            with self.times.run('decode get_tau'):
                tau = read_tau(fin, out_shape=(3, self.K, 1, 1), device=l.sigmas.device)
                l = prob_clf.copy_network_output(l, sigmas=l.sigmas * tau)

        x_range = read_range(fin)
        C, H, W = read_shapes(fin)
        r = ArithmeticCoder(L=x_range[1] - x_range[0] + 1)

        # We decode channel by channel, see `encode_scale`.
        def decoder(_, C_cur):
            num_bytes = read_num_bytes_encoded(fin)
            encoded = fin.read(num_bytes)
            residual_sym_truncated_raw_c = r.range_decode(encoded, cdf=C_cur, time_logger=self.times).reshape(1, H, W)
            residual_sym_truncated_raw_c = residual_sym_truncated_raw_c.to(pe.DEVICE, non_blocking=True)
            # NOTE: here it's int16
            bn_c = SymbolTensor(residual_sym_truncated_raw_c + x_range[0], L=511, centered=True).to_norm().t
            # yielding always bottleneck and extra_info (=None here)
            return bn_c, None

        with self.times.prefix_scope('decode scale'):
            with self.times.run('total'):
                bn, _ = self.code_with_cdf(l, (1, C, H, W), decoder, dmll, x_range)

        return bn

    def code_with_cdf(self, l: prob_clf.NetworkOutput, bn_shape, bn_coder, dmll, x_range):
        """
        :param l: predicted distribution, i.e., NKpHW, see DiscretizedMixLogisticLoss
        :param bn_shape: shape of the bottleneck to encode/decode
        :param bn_coder: function with signature (c: int, C_cur: CDFOut) -> (bottleneck[c], extra_info_c). This is
        called for every channel of the bottleneck, with C_cur == CDF to use to encode/decode the channel. It shoud
        return the bottleneck[c].
        :param dmll: instance of DiscretizedMixLogisticLoss
        :return: decoded bottleneck (in [-1, 1], list of all extra info produced by `bn_coder`.
        """
        N, C, H, W = bn_shape
        coding = bitcoding.coders_helpers.CodingCDFNonshared(l, dmll=dmll, x_range=x_range, centered_x=True)

        # needed also while encoding to get next C
        decoded_bn = torch.zeros(N, C, H, W, dtype=torch.float32).to(pe.DEVICE)
        extra_info = []

        with self.times.combine('c{} {:.5f}'):
            for c in range(C):
                with self.times.run('get_C'):
                    C_cond_cur = coding.get_next_C(decoded_bn)
                with self.times.run('bn_coder'):
                    decoded_bn[:, c, ...], extra_info_c = bn_coder(c, C_cond_cur)
                    extra_info.append(extra_info_c)

        return decoded_bn, extra_info


def _get_cdf_from_pr(pr):
    """
    :param pr: NHWL
    :return: NHW(L+1) as int16 on CPU!
    """
    N, H, W, _ = pr.shape

    precision = 16

    cdf = torch.cumsum(pr, -1)
    cdf = cdf.mul_(2**precision)
    cdf = cdf.round()
    cdf = torch.cat((torch.zeros((N, H, W, 1), dtype=cdf.dtype, device=cdf.device),
                     cdf), dim=-1)
    cdf = cdf.to('cpu', dtype=torch.int16, non_blocking=True)

    return cdf


def _get_uniform_pr(S_shape, L):
    N, C, H, W = S_shape
    assert N == 1
    histo = torch.ones(L, dtype=torch.float32) / L
    assert (1 - histo.sum()).abs() < 1e-5, (1 - histo.sum()).abs()
    extendor = torch.ones(N, H, W, L)
    pr = extendor * histo
    return pr.to(pe.DEVICE)


def write_tau(tau, fout):
    tau = tau.flatten().detach().cpu().tolist()
    write_bytes(fout, [np.float32 for _ in range(len(tau))], tau)
    return len(tau) * 4


def read_tau(fin, out_shape, device):
    tau = read_bytes(fin, [np.float32 for _ in range(np.prod(out_shape))])
    return torch.tensor(tau).reshape(out_shape).to(device)


def write_range(x_range, fout):
    mi, ma = x_range
    assert mi >= -255 and ma <= 255
    write_bytes(fout, [np.int16, np.int16], x_range)
    return 4


def read_range(fin):
    return tuple(map(int, read_bytes(fin, [np.int16, np.int16])))


def write_shape(shape, fout):
    """
    Write tuple (C,H,W) to file, given shape 1CHW.
    :return number of bytes written
    """
    assert len(shape) == 4 and shape[0] == 1, shape
    shape = shape[1:]
    assert shape[0] < 2**8,  shape
    assert shape[1] < 2**16, shape
    assert shape[2] < 2**16, shape
    assert len(shape) == 3,  shape
    write_bytes(fout, [np.uint8, np.uint16, np.uint16], shape)
    return 5


def read_shapes(fin):
    return tuple(map(int, read_bytes(fin, [np.uint8, np.uint16, np.uint16])))


def write_num_bytes_encoded(num_bytes, fout):
    assert num_bytes < 2**32
    write_bytes(fout, [np.uint32], [num_bytes])
    return 2  # number of bytes written


def read_num_bytes_encoded(fin):
    return int(read_bytes(fin, [np.uint32])[0])


def write_bytes(f, ts, xs):
    for t, x in zip(ts, xs):
        f.write(t(x).tobytes())


@ft.return_list
def read_bytes(f, ts):
    for t in ts:
        num_bytes_to_read = t().itemsize
        yield np.frombuffer(f.read(num_bytes_to_read), t, count=1)


# ---


def test_write_shapes(tmpdir):
    p = str(tmpdir.mkdir('test').join('hi.l3c'))
    with open(p, 'wb') as f:
        write_shape((1,2,3), f)
    with open(p, 'rb') as f:
        assert read_shapes(f) == (1,2,3)


def test_write_bytes(tmpdir):
    p = str(tmpdir.mkdir('test').join('hi.l3c'))
    with open(p, 'wb') as f:
        write_num_bytes_encoded(1234567, f)
    with open(p, 'rb') as f:
        assert read_num_bytes_encoded(f) == 1234567


def test_bytes(tmpdir):
    shape = (3, 512, 768)
    p = str(tmpdir.mkdir('test').join('hi.l3c'))
    with open(p, 'wb') as f:
        write_bytes(f, [np.uint8, np.uint16, np.uint16], shape)
    with open(p, 'rb') as f:
        c, h, w = read_bytes(f, [np.uint8, np.uint16, np.uint16])
        assert (c, h, w) == shape
