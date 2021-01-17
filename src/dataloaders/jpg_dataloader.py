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

import numpy as np
import os
import random
import argparse

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import RandomCrop

from dataloaders import images_loader
from dataloaders.cached_listdir_imgs import cached_listdir_imgs


from torchvision.transforms import functional as F

from helpers.global_config import global_config


def get_residual_dataset_jpg(raw_images_dir, random_transforms: bool,
                             random_scale, crop_size: int, max_imgs=None,
                             discard_shitty=True,
                             #filter_min_size=None, top_only=None,
                             #is_training=False, sort=False
                             ):
    imgs = cached_listdir_imgs(raw_images_dir, min_size=512, discard_shitty=discard_shitty).sort()
    return JPGDataset(imgs,
                      random_crops=crop_size if random_transforms else None,
                      random_flips=random_transforms,
                      random_scale=random_scale,
                      center_crops=crop_size if not random_transforms else None,
                      max_imgs=max_imgs)

class JPGDataset(Dataset):
    def __init__(self, images,
                 random_crops=None,
                 random_flips=False,
                 random_scale=False,
                 center_crops=None,
                 max_imgs=None):
        self.random_crops = random_crops
        self.random_flips = random_flips
        self.random_scale = random_scale
        self.center_crops = center_crops
        self.imgs = images.ps
        self.id = f'C{images.id}_JPG'
        self.to_tensor_transform = images_loader.IndexImagesDataset.to_tensor_uint8_transform()
        self.quality = 97

    def get_filename(self, idx):
        return os.path.splitext(os.path.basename(self.imgs[idx]))[0]

    @staticmethod
    def _random_scale_params(im, scale_min=0.4, scale_max=1.0, size_min=None):
        W, H = im.size
        D = min(W, H)
        if size_min:
            # random scale has to be >=scale_min to full fill min(output.size) >= size_min
            scale_min = size_min / D
        scale = random.uniform(scale_min, scale_max)
        W2 = round(W * scale)
        H2 = round(H * scale)
        return W2, H2

    def __len__(self):
        return len(self.imgs)

    def _get_jpg(self, im):
        tmp_dir = os.path.join('/dev/shm', str(os.getpid()))
        os.makedirs(tmp_dir, exist_ok=True)

        img_p = os.path.join(tmp_dir, 'img.jpg')
        quality = self.quality
        if global_config.get('rand_quality'):
            quality = random.randint(95, 99)
        im.save(img_p, quality=quality)

        bpp = os.path.getsize(img_p) * 8 / np.prod(im.size)

        return Image.open(img_p).convert('RGB'), bpp

    def write_file_names_to_txt(self, log_dir):
        assert os.path.isdir(log_dir)
        with open(os.path.join(log_dir, 'train_imgs.txt'), 'w') as fout:
            fout.write(f'{len(self.imgs)} raw_images, {self.id}\n---\n')
            fout.write('\n'.join(map(os.path.basename, self.imgs)))

    def __getitem__(self, idx):
        p = self.imgs[idx]
        im, f = JPGDataset._read_img(p)

        if self.random_scale:
            im, = JPGDataset.random_scale_imgs(im, size_min=512)
        if self.random_crops:
            im, = JPGDataset.crop_imgs(im, size=(self.random_crops, self.random_crops))
        if self.center_crops:
            im, = JPGDataset.center_crop_imgs(im, size=(self.center_crops, self.center_crops))
        im = JPGDataset._convert(im)

        im_jpg, bpp = self._get_jpg(im)

        im = self.to_tensor_transform(im)
        im_jpg = self.to_tensor_transform(im_jpg)

        out = {'bpp': bpp}

        f.close()

        out['raw'] = im
        out['compressed'] = im_jpg

        return out

    @staticmethod
    def random_scale_imgs(*pil_imgs, size_min):
        # TODO: not sure if it makes sense to rescale BPG...
        W_out, H_out = JPGDataset._random_scale_params(pil_imgs[0], size_min=size_min)
        for img in pil_imgs:
            yield img.resize((W_out, H_out), resample=Image.LANCZOS)

    @staticmethod
    def center_crop_imgs(*pil_imgs, size):
        for img in pil_imgs:
            yield F.center_crop(img, size)

    @staticmethod
    def crop_imgs(*pil_imgs, size):
        i, j, h, w = RandomCrop.get_params(pil_imgs[0], size)
        for img in pil_imgs:
            yield F.crop(img, i, j, h, w)

    @staticmethod
    def _convert(pil_img):
        if pil_img.mode != 'RGB':
            return pil_img.convert('RGB')
        return pil_img

    @staticmethod
    def _read_img(p):
        f = open(p, 'rb')
        # with  as f:
        return Image.open(f), f

