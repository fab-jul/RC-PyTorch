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
from contextlib import contextmanager

import pytorch_ext as pe

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from helpers.notebook_dict import get_is_in_notebook


class _ClassesCache(object):
    def __init__(self):
        self.current = None  # N x nB


_classes = _ClassesCache()

_all_vars = {'gamma': [], 'beta': []}

@contextmanager
def cin_classes_context(flag, classes_maker, bpps):
    if flag:
        set_cin_classes(classes_maker(bpps))
    yield
    if flag:
        set_cin_classes(None)


def set_cin_classes(classes):
    _classes.current = classes.detach() if classes is not None else None


class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, Cf, nB, eps=1e-5, momentum=0.1):
        super().__init__()
        self.Cf = Cf
        self.nB = nB
        # self.instance_norm = nn.InstanceNorm2d(Cf, affine=False)
        self.eps = eps
        self.momentum = momentum
        # nB x Cf
        self.gamma = nn.Parameter(torch.Tensor(nB, Cf))
        # nB x Cf
        self.beta = nn.Parameter(torch.Tensor(nB, Cf))

        # TODO: for debugging
        if get_is_in_notebook():
            _all_vars['gamma'].append(self.gamma)
            _all_vars['beta'].append(self.beta)

        self.reset_parameters()

    # following BN2d implementationg
    def reset_parameters(self):
        init.uniform_(self.gamma)
        init.zeros_(self.beta)

    def get_params_for_classes(self, classes=None):
        if classes is None:
            classes = _classes.current
        # classes expected to be N x nB
        assert classes is not None
        assert len(classes.shape) ==2 and classes.shape[1] == self.nB, classes.shape  # for every batch please

        # N x Cf x 1 x 1
        gamma = torch.matmul(classes, self.gamma).unsqueeze(-1).unsqueeze(-1)
        # N x Cf x 1 x 1
        beta = torch.matmul(classes, self.beta).unsqueeze(-1).unsqueeze(-1)

        return gamma, beta


    def forward(self, x, classes=None):
        gamma, beta = self.get_params_for_classes(classes)
        x = F.instance_norm(
                x,
                running_mean=None, running_var=None,
                weight=None, bias=None,
                use_input_stats=True,
                momentum=self.momentum, eps=self.eps)
        return gamma * x + beta
        # return F.instance_norm(
        #         x, self.running_mean, self.running_var, self.weight, self.bias,
        #         self.training or not self.track_running_stats, self.momentum, self.eps)


def test_cin2d():
    Cf = 3
    nB = 5
    N = 4
    c = ConditionalInstanceNorm2d(Cf, nB)

    x = torch.randint(0, nB, (N,))
    print(x)
    x = pe.one_hot(x, L=nB, Ldim=-1)
    print(x)

    i = torch.rand(N, Cf, 10, 10)

    with set_cin_classes(x):
        y = c(i)

