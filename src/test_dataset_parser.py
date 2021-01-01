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
import os
import re

from blueprints import shared
from dataloaders.compressed_images_loader import get_residual_dataset, \
    MetaResidualDataset
from dataloaders.images_loader import IndexImagesDataset
from helpers.testset import Testset


def name_from_images(flag):
    if ';' in flag:
        raw, compressed = flag.split(';')
        raw_dataset_name = os.path.basename(raw.rstrip(os.path.sep))  # e.g.  test_new_clean
        compressed_dataset_name = os.path.basename(compressed.rstrip(os.path.sep))  # e.g. test_new_clean_bpg_q12
        return raw_dataset_name, compressed_dataset_name
    elif '_multi_q' in flag:
        raw, qs = flag.split('_multi_q')
        assert os.path.isdir(raw), raw
        raw = raw.rstrip(os.path.sep)
        compressed_dataset_name = os.path.basename(flag).rstrip(os.path.sep)
        return raw, compressed_dataset_name
    raise ValueError(flag)

def make_ds(raw, qs_to_dirs, crop, max_imgs_per_folder):
    return {q: get_residual_dataset(
                  {'raw': raw, 'compressed': compressed},
                  random_transforms=False, random_scale=False,
                  crop_size=crop, mode='both', max_imgs=max_imgs_per_folder,
                  discard_shitty=False, sort=True)
            for q, compressed in qs_to_dirs}


def parse_into_datasets(images, crop=None, match_filenames=None, max_imgs_per_folder=None):
    # Append flags.crop to ID so that it creates unique entry in cache
    append_id = f'_crop{crop}' if crop else None

    images = images.split(',')

    # check all are the same kind
    kinds = ['enhancement' if ';' in i else 'multiscale' for i in images]
    assert len(set(kinds)) == 1, f'Expected all same kind, got {kinds}'

    for images_dir_or_image in images:
        if ';' in images_dir_or_image:
            if match_filenames:
                raise NotImplementedError

            # TODO: max_imgs_per_folder should affect ID
            raw, compressed = images_dir_or_image.split(';')

            yield get_residual_dataset({'raw': raw, 'compressed': compressed}, random_transforms=False,
                                       random_scale=False,
                                       crop_size=crop, mode='both', max_imgs=max_imgs_per_folder, discard_shitty=False,
                                       sort=True)  # TODO
        elif images_dir_or_image.startswith('AUTOEXPAND:'):
            if match_filenames:
                raise NotImplementedError

            # has form AUTOEXPAND:/path/to/some/folder_of_raws,
            # and /path/to/some/folder_of_raws_bpg_q* exist
            raw = images_dir_or_image.replace('AUTOEXPAND:', '').rstrip('/')
            assert os.path.isdir(raw), raw
            bpgs_glob = raw + '_bpg_q*'
            bpgs_dirs = glob.glob(bpgs_glob)
            assert len(bpgs_dirs) >= 1, \
                f'No BPG dirs found. Did you preprocess? {bpgs_glob}'
            get_q_re = re.compile(r'_bpg_q(\d+)$')
            try:
                qs_to_dirs = sorted((int(get_q_re.search(bpg_dir).group(1)), bpg_dir)
                                    for bpg_dir in bpgs_dirs)
            except AttributeError: # If the match fails
                raise ValueError(f'Error parsing BPG dirs: {bpgs_dirs}')
            print('*** AUTOEXPAND ->\n' +
                  '\n'.join(f'{q}: {p}' for q, p in qs_to_dirs))

            ds = make_ds(raw, qs_to_dirs, crop, max_imgs_per_folder)
            yield MetaResidualDataset(ds, name=os.path.basename(images_dir_or_image).rstrip('/'))
        elif '_multi_q' in images_dir_or_image:   # Legacy
            if match_filenames:
                raise NotImplementedError

            # has form /path/to/some/folder_of_raws_multi_q12_13_15_17
            raw, qs = images_dir_or_image.split('_multi_q')
            assert os.path.isdir(raw), raw
            raw = raw.rstrip(os.path.sep)
            qs = map(int, qs.split('_'))
            qs_to_dirs = [(q, compressed + '_bpg_q' + str(q)) for q in qs]
            assert all(
              os.path.isdir(compressed) for _, compressed in qs_to_dirs), \
                qs_to_dirs

            ds = make_ds(raw, qs_to_dirs, crop, max_imgs_per_folder)
            yield MetaResidualDataset(ds, name=os.path.basename(images_dir_or_image).rstrip('/'))

        else:
            raise NotImplementedError
