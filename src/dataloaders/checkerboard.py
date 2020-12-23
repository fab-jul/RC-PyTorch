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
import torch

from dataloaders.dataloader_helpers import RepeatDataset
import numpy as np

from PIL import Image
import os, shutil

def make_checkerboard_dataset(crop_size, values, pattern_sizes, num_els):
    return RepeatDataset(_CheckerboardDataset(crop_size, values, pattern_sizes), num_els)


class _CheckerboardDataset(object):
    def __init__(self, crop_size, values, pattern_sizes):
        assert crop_size % 2 == 0, f'Must be divisible by 2! {crop_size}'

        possible_boards = list(_possible_boards(values))

        n = len(values)
        assert len(possible_boards) == n * (n - 1)
        assert len(set(possible_boards)) == len(possible_boards)
        print('*** # possible_boards = ', n * (n - 1))

        self.pat = [
            torch.from_numpy(_generate_pattern(b, crop_size, pattern_size))
            for b in possible_boards
            for pattern_size in pattern_sizes]

        # print('Patterns')
        # print(a[0, :5, :5])
        # print(a[1, :5, :5])
        # print(a[2, :5, :5])
        # print(b[0, :5, :5])
        # print(a[1, :5, :5])
        # print(b[2, :5, :5])
        # print(a.dtype, b.dtype)
        # print('-' * 80)

    def __len__(self):
        return len(self.pat)

    def __getitem__(self, idx):
        return self.pat[idx]

    def __str__(self):
        return 'CheckerboardDataset()'

    def save_all(self, out_dir):
        dirout = os.path.join(out_dir, 'checkerboards')
        print(f'Saving {len(self)} images in {dirout}...')
        if os.path.isdir(dirout):
            shutil.rmtree(dirout)
        os.makedirs(dirout)

        for i in range(len(self.pat)):
            p = self.pat[i]
            Image.fromarray(p.detach().permute(1, 2, 0).cpu().numpy()).save(os.path.join(dirout, f'{i:010}.png'))


def _possible_boards(values):
    for i, vi in enumerate(values):
        for j, vj in enumerate(values):
            if vi == vj:
                continue
            yield vi, vj


def _generate_pattern(values, crop_size, cell_width):
    assert (crop_size // 2) % cell_width == 0

    tile_1 = np.tile(values[0], (cell_width, cell_width)).astype(np.uint8)
    tile_2 = np.tile(values[1], (cell_width, cell_width)).astype(np.uint8)
    tile_12 = np.hstack((tile_1, tile_2))
    tile_21 = np.hstack((tile_2, tile_1))
    tile = np.vstack((tile_12, tile_21))

    num_cells = crop_size // 2 // cell_width
    pat = np.tile(tile, (num_cells, num_cells))
    pat_img = np.stack([pat, pat, pat], 0)
    return pat_img


def test_checkerboard():
    d = make_checkerboard_dataset(8, [10, 20], [1], 30)
    for i in range(10):
        print(d[i][0, ...])

def test_checkerboard_2():

    for n in range(2, 5):
        for pattern_sizes in [[1], [1,2], [1, 2, 4]]:
            vals = list(np.linspace(10, 240, n))
            print(vals)
            d = _CheckerboardDataset(8, vals, pattern_sizes)
            d.save_all(os.path.expanduser(f'~/ckpts/{n}_{"".join(map(str, pattern_sizes))}'))

