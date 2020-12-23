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
import os

from torch.utils.data.dataset import Dataset

from dataloaders.cached_listdir_imgs import cached_listdir_imgs, Images
from dataloaders.images_loader import IndexImagesDataset
from test.test_helpers import QHistory


class ClassifierDataset(Dataset):
    """
    uses the intersection of (images in images_dir) and (images specified by optimal_qs)
    """
    def __init__(self, images_dir, optimal_qs_csv, to_tensor_transform):
        images = cached_listdir_imgs(images_dir, discard_shitty=False)
        self.optimal_qs = QHistory.read_q_history(optimal_qs_csv)
        assert self.optimal_qs, optimal_qs_csv
        print('Read optimal Q for', len(self.optimal_qs), 'images.')
        missing = {p for p in images.ps
                   if os.path.splitext(os.path.basename(p))[0] not in self.optimal_qs}
        if missing:
            print(f'Missing files in {optimal_qs_csv}: {len(missing)}')
        image_ps = list(set(images.ps) - missing)
        assert len(image_ps) > 0, (images_dir, optimal_qs_csv)
        print(f'--> Using {len(image_ps)} images!')
        self.images_ds = IndexImagesDataset(
                Images(image_ps, images.id), to_tensor_transform)
        self.id = f'{images.id}_Q{os.path.basename(optimal_qs_csv)}'

    def get_filename(self, idx):
        return os.path.splitext(os.path.basename(self.images_ds.files[idx]))[0]

    def __getitem__(self, idx):
        p = self.images_ds.files[idx]
        fn = os.path.splitext(os.path.basename(p))[0]
        q = self.optimal_qs[fn]
        return {'raw': self.images_ds.load_transform_img(p), 'q': q}

    def __len__(self):
        return len(self.images_ds)

    def write_file_names_to_txt(self, log_dir):
        assert os.path.isdir(log_dir)
        with open(os.path.join(log_dir, 'train_imgs.txt'), 'w') as fout:
            fout.write(f'{len(self.images_ds.files)} compressed_images, {self.id}\n---\n')
            fout.write('\n'.join(map(os.path.basename, self.images_ds.files)))
