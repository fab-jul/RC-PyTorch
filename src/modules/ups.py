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
from torch import nn


class DeconvUp(nn.Module):
    def __init__(self, config_ms):
        super(DeconvUp, self).__init__()
        Cf = config_ms.Cf
        kernel_size = 4  # even for less checkerboards, #fingerscrossed
        self.conv_transpose = nn.ConvTranspose2d(Cf, Cf, kernel_size,
                                                 stride=2, padding=1, output_padding=0,
                                                 bias=True)

    def forward(self, x):
        return self.conv_transpose(x)

    def __repr__(self):
        return 'DeconvUp()'


class ConvResizeConvUp(nn.Module):
    def __init__(self, config_ms):
        super(ConvResizeConvUp, self).__init__()
        Cf = config_ms.Cf
        kernel_size = 3
        self.conv = nn.Conv2d(Cf, 2*Cf, kernel_size, padding=kernel_size//2)
        self.up = nn.Conv2d(2*Cf, 2*Cf, kernel_size, padding=kernel_size//2)
        self._repr = f'ConvResizeConvUp({2*Cf})'

    def __repr__(self):
        return self._repr

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        return self.up(x)


class ResizeConvUp(nn.Module):
    def __init__(self, config_ms):
        super(ResizeConvUp, self).__init__()
        Cf = config_ms.Cf
        kernel_size = 3
        self.up = nn.Conv2d(Cf, Cf, kernel_size, padding=kernel_size//2)
        self._repr = f'ResizeConvUp({Cf})'

    def __repr__(self):
        return self._repr

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        return self.up(x)