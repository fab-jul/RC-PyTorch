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
from torch.utils.data.dataset import Dataset


class RepeatDataset(Dataset):
    def __init__(self, ds, num_els):
        assert num_els > len(ds)
        self.ds = ds
        self.num_els = num_els

    def __len__(self):
        return self.num_els

    def __getitem__(self, idx):
        return self.ds[idx % len(self.ds)]

    def __str__(self):
        return f'RepeatDataset({str(self.ds)}, num_els={self.num_els})'



def test_repeat():
    class IntDataset(Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, item):
            return item

    r = RepeatDataset(IntDataset(), 10)
    for i in range(len(r)):
        print(r[i])
