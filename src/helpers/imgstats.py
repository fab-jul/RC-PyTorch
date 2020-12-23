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
import numpy as np


_INTERNAL_OFFSET = 256


def truncate_histo(histo, min_val=1e-9):
    mi = histo.min()
    start, stop = None, None
    for i, v in enumerate(histo):
        if abs(v - mi) < min_val:
            if start is not None:
                stop = i
                break
        else:
            if start is None:
                start = i
    return np.arange(start-_INTERNAL_OFFSET, stop-_INTERNAL_OFFSET), histo[start:stop]


def avg_histo(res, stop=None):
    res = iter(res)
    histos = histo_single_image(next(res))
    for i, r in enumerate(res):
        if stop and i >= stop:
            break
        histos_res = histo_single_image(r)
        for h, h_r in zip(histos, histos_res):
            h += h_r
    return histos


def histo_single_image(res):
    minlength = 256 * 2
    res = res + _INTERNAL_OFFSET
    histos = [np.bincount(res[..., c].ravel(), minlength=minlength)
              for c in range(3)]
    mmax = max(np.max(h) for h in histos)
    histos = [h / mmax for h in histos]
    return histos

