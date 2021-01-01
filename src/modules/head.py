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
from modules.gdn import GDN
from pytorch_ext import default_conv as conv


class RGBHead(nn.Module):
    """ Go from 3 channels (RGB) to Cf channels, also normalize RGB """
    def __init__(self, config_ms):
        super(RGBHead, self).__init__()
        assert 'Subsampling' not in config_ms.enc.cls, 'For Subsampling encoders, head should be ID'
        head = [
            Head(config_ms, Cin=3)
        ]
        if global_config.get('gdn', False):
            print('*** Adding GDN')
            head.append(GDN(config_ms.Cf))
        self.head = nn.Sequential(
                # Note, this actually shifts data to ~[-1, 1]
                # edsr.MeanShift(0, (0., 0., 0.), (128., 128., 128.)),
                *head)
        self._repr = 'MeanShift//Head(C=3)'

    def __repr__(self):
        return f'RGBHead({self._repr})'

    def forward(self, x):
        return self.head(x)


class Head(nn.Module):
    """
    Go from Cin channels to Cf channels.
    For L3C, Cin=Cf, and this is the convolution yielding E^{s+1}_in in Fig. 2.

    """
    def __init__(self, config_ms, Cin):
        super(Head, self).__init__()
        assert 'Subsampling' not in config_ms.enc.cls, 'For Subsampling encoders, head should be ID'
        self.head = conv(Cin, config_ms.Cf, config_ms.kernel_size)
        self._repr = f'Conv({config_ms.Cf})'

    def __repr__(self):
        return f'Head({self._repr})'

    def forward(self, x):
        return self.head(x)


