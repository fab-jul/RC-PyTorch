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
import argparse
import pickle
import os
import random

import fasteners
import numpy as np
import torch
from PIL import Image
from decorator import contextmanager
from fjcommon import timer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as F

from multiprocessing import Manager

import pytorch_ext as pe
from compressor import Compressor
from dataloaders import images_loader
from dataloaders.cached_listdir_imgs import Images, cached_listdir_imgs, cached_listdir_imgs_max

# class CompressedImagesDataset(IndexImagesDataset):
#     def __getitem__(self, idx):
#         p = self.files[idx]
#         d = self.dict_from_path(idx, p)
#         d['bpp'] = Compressor.bpp_from_compressed_file(p)
#         return d
from dataloaders.jpg_dataloader import JPGDataset
from helpers.global_config import global_config


NO_ERRORS = int(os.environ.get('NO_ERRORS', 0)) == 1


def get_residual_dataset(imgs_dir, random_transforms: bool,
                         random_scale, crop_size: int, mode: str, max_imgs=None,
                         discard_shitty=True, filter_min_size=None, top_only=None,
                         is_training=False, sort=False):
    if top_only:
        assert top_only < 1
    multiple_ds = False
    if isinstance(imgs_dir, dict):
        assert 'raw' in imgs_dir and 'compressed' in imgs_dir, imgs_dir.keys()
        raw_p, compressed_p = imgs_dir['raw'], imgs_dir['compressed']
        multiple_ds = isinstance(imgs_dir['raw'], list)
        if multiple_ds:
            assert len(raw_p) == len(compressed_p)
    elif ';' in imgs_dir:
        raw_p, compressed_p = imgs_dir.split(';')
    else:
        raise ValueError('Invalid imgs_dir, should be dict or string with ;, got', imgs_dir)

    # works fine if p_ is a list
    get_imgs = lambda p_: cached_listdir_imgs(
            p_, min_size=filter_min_size or crop_size, discard_shitty=discard_shitty)

    if compressed_p == 'JPG':
        print('*** Using JPG...')
        imgs = get_imgs(raw_p)
        return JPGDataset(imgs,
                          random_crops=crop_size if random_transforms else None,
                          random_flips=random_transforms,
                          random_scale=random_scale,
                          center_crops=crop_size if not random_transforms else None,
                          max_imgs=max_imgs)

    if is_training and global_config.get('filter_imgs', False):
        assert not multiple_ds
        print('*** filtering', imgs_dir)
        filter_imgs = global_config['filter_imgs']
        if not isinstance(filter_imgs, int):
            filter_imgs = 680
        print(filter_imgs)
        get_imgs = lambda p_: cached_listdir_imgs_max(p_, max_size=filter_imgs, discard_shitty=True)

    raw, compressed = map(get_imgs, (raw_p, compressed_p))

    if top_only:
        sorted_imgs = sorted((Compressor.bpp_from_compressed_file(p), p) for p in compressed.ps)
        top_only_imgs = sorted_imgs[-int(top_only * len(sorted_imgs)):]
        top_only_ps = [p for _, p in top_only_imgs]
        compressed = Images(top_only_ps, compressed.id + f'_top{top_only:.2f}')
        print(f'*** Using {len(top_only_ps)} of {len(sorted_imgs)} images only')

    if sort:
        print('Sorting...')
        raw = raw.sort()
        compressed = compressed.sort()

    return ResidualDataset(compressed, raw,
                           mode=mode,
                           random_crops=crop_size if random_transforms else None,
                           random_flips=random_transforms,
                           random_scale=random_scale,
                           center_crops=crop_size if not random_transforms else None,
                           max_imgs=max_imgs)


def _filter_big_images(raw, compressed, max_num_pixels=2*512*512):
    def _filter(p):
        i = Image.open(p)
        if np.prod(i.size) > max_num_pixels:
            print(f'*** Filtering {p}, too many pixels {i.size}')
            return False
        return True

    raw = raw.filter(_filter)
    compressed = compressed.filter(_filter)
    return raw, compressed


def is_residual_dataset(o):
    return isinstance(o, ResidualDataset) or isinstance(o, MetaResidualDataset)


# class _Cache(object):
#     def __init__(self, root_dir, name):
#         self.root_dir = root_dir
#         self.lock_file = os.path.join(root_dir, f'optimal_q_cache_{name}_lock')
#         self.cache_file = os.path.join(root_dir, f'optimal_q_cache_{name}.pkl')
#
#     def read(self):
#         with fasteners.InterProcessLock(self.lock_file):
#             if not os.path.isfile(self.cache_file):
#                 return {}
#             with open(self.cache_file, 'rb'):
#                 return pickle.load(self.cache_file)
#
#     def write(self, filename_to_q):
#         with fasteners.InterProcessLock(self.lock_file):
#             cache_file_tmp = self.cache_file + '_tmp'
#             with open(self.cache_file, 'wb'):


class MetaResidualDataset(Dataset):
    def __init__(self, datasets, name):
        """
        :param datasets: dict q -> ResidualDataset
        """
        self.name = name
        assert len(set(map(len, datasets.values()))) == 1, list(map(len, datasets.values()))
        self.datasets = datasets
        self.some_dataset: ResidualDataset = next(iter(self.datasets.values()))
        qs = '_'.join(map(str, datasets.keys()))
        self.id = f'{self.some_dataset.raw_images_id}_multi_q{qs}'
        self.starting_q = 13 if 13 in datasets.keys() else sorted(datasets.keys())[len(datasets.keys())//2]
        print('Starting Q', self.starting_q)
        self.idx_to_skip = None

    def get_min_bpp_img(self):
        _, idx = min(self.some_dataset._iter_bpp_idx())
        return self[idx]

    def get_max_bpp_img(self):
        _, idx = max(self.some_dataset._iter_bpp_idx())
        return self[idx]

    def get_raw_p(self, idx):
        return self.some_dataset.get_raw_p(idx)

    def set_skip(self, names_to_skip, modulo_op):
        if names_to_skip:
            self.idx_to_skip = {idx for idx in range(len(self))
                                if self.some_dataset.get_filename(idx) in names_to_skip
                                or (modulo_op and (idx % 3 != (modulo_op - 1)))}
            print('Skipping', len(self.idx_to_skip), 'files!!!')

    def __len__(self):
        return len(self.some_dataset)

    def __getitem__(self, idx):
        """ :return dictionary {q -> ds_q[idx]} """
        if self.idx_to_skip and idx in self.idx_to_skip:
            return None
        return {q: ds[idx] for q, ds in self.datasets.items()}

    def get_filename(self, idx):
        filenames = {ds.get_filename(idx) for ds in self.datasets.values()}
        assert len(filenames) == 1, filenames
        return filenames.pop()

    def get_bpg_bpsps(self, filenames_Qs):
        # Dict Q -> (raw_fn -> compressed_fn)
        q_to_d = {Q: self.datasets[Q].get_raw_to_compressed_dict()
                  for Q in self.datasets.keys()}
        return [Compressor.bpp_from_compressed_file(q_to_d[Q][fn])/3 for fn, Q in filenames_Qs]

    def iter_residuals(self, filenames_Qs):
        # Dict Q -> (raw_fn -> compressed_fn)
        q_to_d = {Q: self.datasets[Q].get_raw_to_compressed_dict()
                  for Q in self.datasets.keys()}
        for fn, Q in filenames_Qs:
            compressed_p = q_to_d[Q][fn]
            raw_p = self.datasets[Q].compressed_to_raw[os.path.basename(compressed_p)]
            res = _open_long(raw_p) - _open_long(compressed_p)
            yield res


def _open_long(p):
    return np.array(Image.open(p)).astype(np.long)



ENABLE_CACHE = int(os.environ.get('ENABLE_CACHE', 0)) == 1

# TODO: probably sane to remove diff mode?
class ResidualDataset(Dataset):
    def get_raw_to_compressed_dict(self):
        """ Get dict raw_fn -> compressed_fn (includes bpsp) """
        get_fn = lambda p: os.path.splitext(os.path.basename(p))[0]
        return {get_fn(self.compressed_to_raw[os.path.basename(c_b)]): c_b
                for c_b in self.compressed_images.ps}

    def __init__(self, compressed_images: Images, raw_images: Images, mode,
                 random_crops=None,
                 random_flips=False,
                 random_scale=False,
                 center_crops=None,
                 output_device=None,
                 max_imgs=None):
        assert mode in ('diff', 'both')
        self._manager = Manager() if ENABLE_CACHE else None
        self._cache = self._manager.dict() if ENABLE_CACHE else None
        self._cache_lock = self._manager.Lock() if ENABLE_CACHE else None
        self.mode = mode
        self.raw_images_id = raw_images.id
        self.raw_root_dir = os.path.dirname(raw_images.ps[0])
        self.id = f'C{compressed_images.id}_{self.raw_images_id}'
        self.random_crop_size = (random_crops, random_crops) if random_crops else None
        self.random_scale = random_scale
        self.center_crop_size = (center_crops, center_crops) if center_crops else None
        assert not (self.random_crop_size and self.center_crop_size), 'Only one can be active'
        self.random_flips = random_flips
        self.output_device = output_device

        # if not all have the same dir: we have multiple ds.
        multiple_ds = len(set(map(os.path.dirname, compressed_images.ps))) > 1
        # TODO this should be a different condition but I'm too tired
        # check_unique = not int(os.environ.get('SKIP_CHECK_UNIQUE', 0))
        # if check_unique and multiple_ds and len(compressed_images.ps) != len(raw_images.ps):
        #     import collections
        #     problems = [(k, v)
        #                 for k, v in collections.Counter(map(os.path.basename, raw_images.ps)).items()
        #                 if v > 1]
        #     if problems:
        #         raise ValueError(f'Non-unique filenames: {problems}')

        self.compressed_images = compressed_images

        self._filter_compressed_images(raw_images)

        if max_imgs and max_imgs < len(self.compressed_images.ps):
            print('Subsampling to use {} imgs of {}...'.format(max_imgs, self))
            idxs = np.linspace(0, len(self.compressed_images.ps) - 1, max_imgs, dtype=np.int)
            ps = np.array(self.compressed_images.ps)[idxs].tolist()
            assert len(ps) == max_imgs
            self.compressed_images = Images(ps, self.compressed_images.id + '_s' + str(max_imgs))
            # print('\n'.join(self.compressed_images.ps))
            # exit()

            self.raw_images_id += f'_m{max_imgs}'
            self.id += f'_m{max_imgs}'

        # basename -> p
        self.compressed_to_raw = self._compressed_to_raw(self.compressed_images, raw_images)
        self.to_tensor_transform = images_loader.IndexImagesDataset.to_tensor_uint8_transform()

        assert len(self) > 0

    def _iter_bpp_idx(self):
        return ((bpp, idx) for idx, bpp in enumerate(map(Compressor.bpp_from_compressed_file,
                                                         self.compressed_images.ps)))

    def get_max_bpp_img(self):
        _, idx = max(self._iter_bpp_idx())
        return self[idx]

    def get_min_bpp_img(self):
        _, idx = min(self._iter_bpp_idx())
        return self[idx]

    def __len__(self):
        return len(self.compressed_images.ps)

    def get_filename(self, idx):
        return Compressor.filename_without_bpp(self.compressed_images.ps[idx])

    def get_raw_p(self, idx):
        p = self.compressed_images.ps[idx]
        return self.compressed_to_raw[os.path.basename(p)]

    def write_file_names_to_txt(self, log_dir):
        assert os.path.isdir(log_dir)
        with open(os.path.join(log_dir, 'train_imgs.txt'), 'w') as fout:
            fout.write(f'{len(self.compressed_images.ps)} compressed_images, {self.id}\n---\n')
            fout.write('\n'.join(map(os.path.basename, self.compressed_images.ps)))

    @staticmethod
    def _convert(pil_img):
        if pil_img.mode != 'RGB':
            return pil_img.convert('RGB')
        return pil_img

    @contextmanager
    def _get_pils(self, idx):
        if self._cache and idx in self._cache:
            yield self._cache[idx]
            return

        compressed_p = self.compressed_images.ps[idx]
        bpp = Compressor.bpp_from_compressed_file(compressed_p)

        raw_p = self.compressed_to_raw[os.path.basename(compressed_p)]

        compressed, f1 = self._read_img(compressed_p)
        raw, f2 = self._read_img(raw_p)
        if not NO_ERRORS:
            assert compressed.size == raw.size, f'Error for {compressed_p}, {raw_p}; {compressed.size, raw.size}'

        if self._cache is not None and idx not in self._cache:
            with self._cache_lock:
                if idx not in self._cache:
                    print('Caching', idx)
                    compressed, raw = map(ResidualDataset._convert, (compressed, raw))
                    self._cache[idx] = compressed, raw, bpp

        yield compressed, raw, bpp

        f1.close()
        f2.close()

    def __getitem__(self, idx):
        # with timer.execute('>>> get'):
        with self._get_pils(idx) as (compressed, raw, bpp):
            try:
                # with timer.execute('>>> pre'):
                if self.random_scale:
                    compressed, raw = self.random_scale_imgs(compressed, raw,
                                                             size_min=min(self.random_crop_size))
                if self.random_crop_size:
                    compressed, raw = self.crop_imgs(compressed, raw, size=self.random_crop_size)
                elif self.center_crop_size:
                    compressed, raw = self.center_crop_imgs(compressed, raw, size=self.center_crop_size)
                if self.random_flips:
                    compressed, raw = self.flip_imgs(compressed, raw)

                # with timer.execute('>>> RGB'):
                compressed, raw = map(ResidualDataset._convert, (compressed, raw))
            except Exception as e:
                print(f'*** Caught {e} for {idx}: {self.get_raw_p(idx)} {self.compressed_images.ps[idx]}')
                return self[0]

            # with timer.execute('>>> read'):
            compressed, raw = map(self.to_tensor_transform, (compressed, raw))

        out = {'bpp': bpp}

        if self.mode == 'diff':
            if self.output_device:
                compressed, raw = compressed.to(self.output_device), raw.to(self.output_device)
            out['raw'] = raw.to(torch.int16) - compressed.to(torch.int16)
        else:
            out['raw'] = raw
            out['compressed'] = compressed

        return out

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

    @staticmethod
    def random_scale_imgs(*pil_imgs, size_min):
        # TODO: not sure if it makes sense to rescale BPG...
        W_out, H_out = ResidualDataset._random_scale_params(pil_imgs[0], size_min=size_min)
        for img in pil_imgs:
            yield img.resize((W_out, H_out), resample=Image.LANCZOS)


    @staticmethod
    def crop_imgs(*pil_imgs, size):
        i, j, h, w = RandomCrop.get_params(pil_imgs[0], size)
        for img in pil_imgs:
            yield F.crop(img, i, j, h, w)

    @staticmethod
    def center_crop_imgs(*pil_imgs, size):
        for img in pil_imgs:
            yield F.center_crop(img, size)

    @staticmethod
    def flip_imgs(*pil_imgs, p=0.5):
        should_flip = random.random() < p
        for img in pil_imgs:
            if should_flip:
                yield F.hflip(img)
            else:
                yield img

    @staticmethod
    def _read_img(p):
        # TODO: in theory, we could have a RAM cache here...
        f = open(p, 'rb')
        # with  as f:
        return Image.open(f), f# .convert('RGB')

    @staticmethod
    def _compressed_to_raw(compressed_images: Images, raw_images: Images):
        """
        Create mapping compressed_image -> raw_image. Can be many to one!
        """
        raw_images_fn_to_p = {os.path.splitext(os.path.basename(p))[0]: p for p in raw_images.ps}
        compressed_to_raw_map = {}

        errors = []
        for p in compressed_images.ps:
            try:
                fn = Compressor.filename_without_bpp(p)
                compressed_to_raw_map[os.path.basename(p)] = raw_images_fn_to_p[fn]
            except KeyError as e:
                errors.append(e)
                if len(errors) > 100:
                    break
                continue
        if errors:
            raise ValueError(f'Missing >={len(errors)} keys:', errors[:10],
                             f'\n{len(compressed_images.ps)} vs {len(raw_images.ps)}')
        return compressed_to_raw_map

    def __str__(self):
        return f'ResidualDataset({len(self.compressed_images.ps)} images, id={self.id})'

    def _filter_compressed_images(self, raw_images):
        """
        Find all compressed images that do not have a raw image, and remove them!
        """
        raw_images_fn_to_p = {os.path.splitext(os.path.basename(p))[0] for p in raw_images.ps}
        missing = set()
        for p in self.compressed_images.ps:
            if Compressor.filename_without_bpp(p) not in raw_images_fn_to_p:
                missing.add(p)
        if missing:
            print(f'*** Missing {len(missing)} files that are in compressed but not in raw.')
            self.compressed_images.ps = [p for p in self.compressed_images.ps if p not in missing]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('raw')
    p.add_argument('compressed', nargs='*')
    flags = p.parse_args()
    if ';' in flags.raw:
        flags.raw, flags.compressed = flags.raw.split(';')
    else:
        flags.compressed = flags.compressed[0]

    raw = cached_listdir_imgs(flags.raw, discard_shitty=False)
    compressed = cached_listdir_imgs(flags.compressed, discard_shitty=False)
    print(raw, compressed)

    print('Average bpp', np.mean([Compressor.bpp_from_compressed_file(p) for p in compressed.ps]))

    r = ResidualDataset(compressed, raw, 'diff', 256, True, output_device=pe.DEVICE)

    # for i in range(len(r)):
    #     print(r[i]['residual'].unique())
    # exit(1)
    d = DataLoader(r, batch_size=10, shuffle=False, num_workers=2)
    mi, ma = None, None
    for b in d:
        res = b['raw']
        # print(res.unique())
        if mi is None:
            mi = res.min()
            ma = res.max()
        else:
            mi = min(res.min(), mi)
            ma = max(res.max(), ma)
        print(mi, ma)



if __name__ == '__main__':
    main()

