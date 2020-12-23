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
import glob
import os
from collections import namedtuple
import torch
from fjcommon import config_parser

from torch import nn

import pytorch_ext as pe

import vis.summarizable_module
from helpers import logdir_helpers
from helpers.global_config import global_config, _GlobalConfig
from helpers.pad import pad
from helpers.quantized_tensor import NormalizedTensor, SymbolTensor
from helpers.saver import Restorer
from modules import gdn
from modules.edsr import ResBlock



class ChannelAverage(nn.Module):
    def forward(self, x):
        return x.mean((2, 3))  # NCHW -> NC


class ClassifierNetwork(vis.summarizable_module.SummarizableModule):
    def __init__(self, config_clf):
        super(ClassifierNetwork, self).__init__()
        self.config_clf = config_clf
        Cf = config_clf.Cf
        num_classes = config_clf.num_classes
        head = config_clf.head
        nB = config_clf.n_resblock
        norm = {'bn': nn.BatchNorm2d,
                'gdn': gdn.GDN,
                'identity': lambda _: pe.IdentityModule()}[config_clf.norm]

        if head == 'down3':
            head = [
                pe.default_conv(3, Cf // 4, 5, stride=2), norm(Cf // 4), nn.ReLU(inplace=True),
                pe.default_conv(Cf // 4, Cf // 2, 5, stride=2), norm(Cf // 2), nn.ReLU(inplace=True),
                pe.default_conv(Cf // 2, Cf, 5, stride=2), norm(Cf), nn.ReLU(inplace=True),
            ]
        elif head == 'down2':
            head = [
                pe.default_conv(3, Cf // 2, 5, stride=2), norm(Cf // 4), nn.ReLU(inplace=True),
                pe.default_conv(Cf // 2, Cf, 5, stride=2), norm(Cf), nn.ReLU(inplace=True),
            ]

        self.head = nn.Sequential(*head)
        norm_cls = lambda: norm(Cf)

        model = [
            ResBlock(pe.default_conv, Cf, kernel_size=3, act=nn.ReLU(inplace=True), norm_cls=norm_cls)
            for _ in range(nB)
        ]

        final_Cf = Cf
        if config_clf.num_res_down == 2:
            model.append(pe.default_conv(Cf, 2*Cf, 5, stride=2))
            norm_cls = lambda: norm(2*Cf)
            model += [
                ResBlock(pe.default_conv, 2*Cf, kernel_size=3, act=nn.ReLU(inplace=True), norm_cls=norm_cls)
                for _ in range(nB)
            ]
            final_Cf = 2*Cf

        if global_config.get('final_conv', False):
            model += [
                pe.default_conv(final_Cf, final_Cf, 3),
                nn.LeakyReLU(inplace=True)
            ]

        self.model = nn.Sequential(
                *model,
                ChannelAverage(),
        )

        if config_clf.deep_tail:
            tail = [nn.Linear(final_Cf, 2*final_Cf), nn.LeakyReLU(inplace=True),
                    nn.Linear(final_Cf, num_classes)]
        else:
            tail = [nn.Linear(final_Cf, num_classes)]
        self.tail = nn.Sequential(
                *tail
        )

    def get_q(self, x):
        assert len(x.shape) == 4 and x.shape[0] == 1, x.shape
        with torch.no_grad():
            q_logits = self.forward(x).q_logits
            _, predicted = torch.max(q_logits, 1)
            return predicted.item() + self.config_clf.first_class

    def forward(self, x):
        self.summarizer.register_images('train', {'input': x}, normalize=True, only_once=True)
        x = self.head(x)
        self.summarizer.register_images('auto', {'after_head': x[0, :3, ...]}, normalize=True, only_once=True)
        x = self.model(x)
        x = self.tail(x)
        return ClassifierOut(q_logits=x)


ClassifierOut = namedtuple('ClassifierOut', ['q_logits'])


def _parse_clf_ckpt_p(clf_ckpt_p):
    assert clf_ckpt_p.endswith('.pt') or clf_ckpt_p.endswith('.pt.tmp'), clf_ckpt_p
    ckpts_p = os.path.dirname(clf_ckpt_p)
    assert ckpts_p.endswith('ckpts'), ckpts_p
    log_dir = os.path.dirname(ckpts_p)
    log_dir_comps = logdir_helpers.parse_log_dir(log_dir, 'configs', ['ms', 'dl'], append_ext='.cf')
    clf_config_p, _ = log_dir_comps.config_paths
    return clf_config_p, log_dir_comps.postfix


def load_classifier(clf_ckpt_p):
    clf_config_p, postfix = _parse_clf_ckpt_p(clf_ckpt_p)
    print(f'Using classifier with config {clf_config_p}')
    clf_config, _ = config_parser.parse(clf_config_p)
    # if postfix:
    #     print('Adding from postfix...', postfix)
    #     c = _GlobalConfig()
    #     c.add_from_flag(postfix)
    #     print('Updaing config with', c)
    #     c.update_config(clf_config)
    clf = ClassifierNetwork(clf_config)
    clf.to(pe.DEVICE)
    print(clf)
    map_location = None if pe.CUDA_AVAILABLE else 'cpu'
    # clf_checkpoint_p = Restorer(ckpts_p).get_latest_ckpt()
    print('Restoring', clf_ckpt_p)
    state_dicts = torch.load(clf_ckpt_p, map_location=map_location)
    clf.load_state_dict(state_dicts['net'])
    print(f'Loaded!')
    return clf


class ClassifierBlueprint(vis.summarizable_module.SummarizableModule):
    def __init__(self, config_clf, is_testing=False):
        super(ClassifierBlueprint, self).__init__()

        self.net = ClassifierNetwork(config_clf)
        self.net = self.net.to(pe.DEVICE)

        self.loss = nn.CrossEntropyLoss()
        self.config_clf = config_clf

        self.padding_fac = self.get_padding_fac()
        print('***' * 10)
        print('*** Padding by a factor', self.padding_fac)
        print('***' * 10)

    @staticmethod
    def get_accuracy(q_logits, q_labels):
        _, predicted = torch.max(q_logits, 1)
        assert predicted.shape == q_labels.shape, (predicted.shape, q_labels.shape)
        return (predicted == q_labels).sum().item() / predicted.shape[0]

    def set_eval(self):
        self.net.eval()
        # self.losses.set_eval()

    def pad(self, x):
        raw, _ = pad(x, self.padding_fac, mode='constant')
        return raw

    def pad_undo(self, x, mode='reflect'):
        raw, undo_pad = pad(x, self.padding_fac, mode=mode)
        return raw, undo_pad

    def forward(self, x: NormalizedTensor) -> ClassifierOut:
        """
        :param xs: tuple of NCHW NormalizedTensor of (raw, lossy)
        :param auto_recurse: int, how many times the last scales should be applied again. Used for RGB Shared.
        :return: layers.multiscale.Out
        """
        return self.net(x.get())

    @staticmethod
    def add_image_summaries(sw, out, global_step, prefix):
        pass  # TODO(enh) maybe

    def get_padding_fac(self):
        return None

    def unpack_batch_pad(self, img_or_imgbatch):
        raw = img_or_imgbatch['raw'].to(pe.DEVICE, non_blocking=True)  # uint8 or int16
        q = img_or_imgbatch['q'].to(pe.DEVICE).view(-1)  # 1d tensor of floats

        if len(raw.shape) == 3:
            raw.unsqueeze_(0)

        if self.padding_fac:
            raw = self.pad(raw)

        q = q - self.config_clf.first_class

        assert len(raw.shape) == 4

        raw = SymbolTensor(raw.long(), L=256)
        return raw.to_norm(), q

    def unpack(self, img_batch, fixed_first=None):
        raw = img_batch['raw'].to(pe.DEVICE, non_blocking=True)  # uint8 or int16
        q = img_batch['q'].to(pe.DEVICE).view(-1)  # 1d tensor of floats

        if fixed_first is not None:
            raw[0, ...] = fixed_first['raw']
            q[0, ...] = fixed_first['q']

        q = q - self.config_clf.first_class

        if self.padding_fac:
            raw = self.pad(raw)

        raw = SymbolTensor(raw.long(), L=256)
        return raw.to_norm(), q

