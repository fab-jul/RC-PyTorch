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
import torch
import torchvision

import vis.grid
from helpers.quantized_tensor import SymbolTensor


def new_bottleneck_summary(s: SymbolTensor):
    """
    Grayscale bottleneck representation: Expects the actual bottleneck symbols.
    :param s: NCHW
    :return: [0, 1] image
    """
    s_raw, L = s.get(), s.L
    assert s_raw.dim() == 4, s_raw.shape
    s_raw = s_raw.detach().float().div(L)
    grid = vis.grid.prep_for_grid(s_raw, channelwise=True)
    assert len(grid) == s_raw.shape[1], (len(grid), s_raw.shape)
    assert [g.max() <= 1 for g in grid], [g.max() for g in grid]
    assert grid[0].dtype == torch.float32, grid.dtype
    return torchvision.utils.make_grid(grid, nrow=5)
