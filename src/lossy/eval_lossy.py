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

import argparse
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from criterion.psnr import np_psnr


def get_image_pairs(in_glob, out_glob):
    eval_glob = lambda g: sorted(glob.glob(g))
    in_ps, out_ps = eval_glob(in_glob), eval_glob(out_glob)
    assert len(in_ps) > 0, in_glob
    assert len(in_ps) == len(out_ps), (in_ps, out_ps)
    return zip(in_ps, out_ps)


def get_stats(in_glob, out_glob):
    stats = defaultdict(list)
    for img_pair in get_image_pairs(in_glob, out_glob):
        try:
            img_i, img_o = map(read_img, img_pair)
        except ValueError as e:
            print(e)
            continue

        stats['psnr'] = np_psnr(img_i, img_o, max_val=255.)
        stats['bpp'] = get_bpp(img_pair[1])
    print_stats(stats)


def print_stats(stats: dict):
    for k, v in sorted(stats.items()):
        print(f'{k}: {np.mean(v):.6f}')


def read_img(img_p):
    img = np.array(Image.open(img_p))
    if img.shape[2] != 3:
        raise ValueError(f'Invalid image, got shape {img.shape} for {img_p}')
    return img


def get_bpp(img_p):
    img = read_img(img_p)
    num_pixels = img.shape[0] * img.shape[1]
    return os.path.getsize(img_p) * 8 / num_pixels




def main():
    p = argparse.ArgumentParser()
    p.add_argument('in_glob')
    p.add_argument('out_glob')

    flags = p.parse_args()
    get_stats(flags.in_glob, flags.out_glob)


if __name__ == '__main__':
    main()