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
import random

from torch.utils.data import Sampler


class PersistentRandomSampler(Sampler):
    def __init__(self, data_source):
        super(PersistentRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.current_epoch = None
        self.num_items_to_skip = 0

    def __iter__(self):
        # Generate the permutation of ALL indices
        all_idxs = list(range(len(self.data_source)))
        # Seed random generagtor with current_epoch
        random.Random(self.current_epoch).shuffle(all_idxs)
        # Throw away the indices to skip
        idxs = all_idxs[self.num_items_to_skip:]
        return iter(idxs)

    def __len__(self):
        return len(self.data_source) - self.num_items_to_skip
