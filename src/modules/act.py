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
from torch import nn

from helpers.global_config import global_config
from modules import gdn


def make(C, inverse):
    return {
        'relu':     lambda: nn.ReLU(True),
        'lrelu':    lambda: nn.LeakyReLU(inplace=True),
        'GDN':      lambda: gdn.GDN(C, inverse=inverse)
    }[global_config.get('act', 'relu')]()

