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
import bisect
import torch
import pytorch_ext as pe
import pickle
import os

import numpy as np

import argparse

from compressor import Compressor
from dataloaders import cached_listdir_imgs


# from oi train Q=12
_DEFAULT_BINS = [0.7616466666666667, 0.9320033333333333, 1.0682099999999999, 1.1955966666666666,
                 1.3245166666666666, 1.4632033333333334, 1.62582, 1.8395400000000002, 2.17753]


def get_num_bins(pkl_p):
    with open(pkl_p, 'rb') as fin:
        return len(pickle.load(fin)) + 1


class Quantizer(object):
    def __init__(self, pkl_p, allow_default=False):
        if not os.path.exists(pkl_p):
            if not allow_default:
                raise FileNotFoundError(pkl_p)
            print('*** WARN: Using default bins')
            self.bin_borders_b = _DEFAULT_BINS
        else:
            with open(pkl_p, 'rb') as fin:
                self.bin_borders_b = pickle.load(fin)
        self.num_bins = len(self.bin_borders_b) + 1

    def quantize(self, bpsp):
        return bisect.bisect_right(self.bin_borders_b, bpsp)

    def quantize_batch_one_hot(self, bpsps):
        assert len(bpsps.shape) == 1, bpsps.shape
        bins = torch.tensor(list(map(self.quantize, bpsps)), device=bpsps.device)
        return pe.one_hot(bins, L=self.num_bins, Ldim=-1)

    def quantize_batch(self, bpsps):
        assert len(bpsps.shape) == 1, bpsps.shape
        return torch.tensor(list(map(self.quantize, bpsps)), device=bpsps.device)

def get_default_pkl_p(img_folder, num_bins):
    if isinstance(img_folder, list):
        img_folder = img_folder[0]
    return img_folder.rstrip(os.path.sep) + f'_bins_nb{num_bins}.pkl'


def make_bin_pkl(img_folder, num_bins, overwrite=False):
    if isinstance(img_folder, list):
        img_folder = img_folder[0]
    pkl_out = get_default_pkl_p(img_folder, num_bins)
    if os.path.isfile(pkl_out):
        return pkl_out

    assert 'train_oi' in img_folder, img_folder  # currently not supported, bc min_size and discard_shitty flags
    ps = cached_listdir_imgs.cached_listdir_imgs(img_folder, min_size=None, discard_shitty=True).ps

    ps_bpsps = sorted(((p, Compressor.bpp_from_compressed_file(p) / 3) for p in ps),
                      key=lambda xy: xy[1])  # sort by bpsp
    bpsps = [bpsp for _, bpsp in ps_bpsps]

    # border     b0    b1     ...   bk         k+1 borders
    # bin_idx  0    1      2  ... k    k+1  => k+2 bins
    #
    # for N bins, we need N-1 borders
    # first border is after 1/NB-th of the data

    # NB + 1 so that we get NB-1 evenly spaced bins *within* the data
    bin_borders_x = np.linspace(0, len(bpsps)-1, num_bins+1, dtype=np.int)
    # throw away the boundaries
    bin_borders_x = bin_borders_x[1:-1]
    bin_borders_b = [bpsps[x] for x in bin_borders_x]

    with open(pkl_out, 'wb') as f:
        print('Saving', bin_borders_b, '\n->', pkl_out)
        pickle.dump(bin_borders_b, f)

    return pkl_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('img_folder')
    p.add_argument('num_bins', type=int)
    flags = p.parse_args()
    make_bin_pkl(flags.img_folder, flags.num_bins)


if __name__ == '__main__':
    main()
