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

--------------------------------------------------------------------------------

This helps make repeated globs of folders with a lot of files fast.
Especially useful on slow distributed filesystems.

cached_glob(p, min_size, discard_shitty)

-> checks if there is p/cached_glob.pkl
    -> if not, create
-> filter all imgs that are < min_size or shitty
-> return


"""


import argparse
import itertools
import multiprocessing
import operator
import os
import pickle
import shutil
import skimage.color
import time
from collections import namedtuple, defaultdict
from typing import Generator

import fasteners
import numpy as np
from PIL import Image
from fjcommon import timer

# for skimage.color
import warnings

import task_array

warnings.filterwarnings("ignore")

PKL_NAME = 'cached_glob.pkl'
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif'}

IMG_DIR_REPL = 'IMG_DIR'


class Images(object):
    def __init__(self, ps, id):
        self.ps = ps
        self.id = id

    def search_path(self):
        return self.id

    def __len__(self):
        return len(self.ps)

    def __repr__(self):
        return f'Images({self.id};{len(self.ps)})'

    def repeat(self, count):
        ps = itertools.chain.from_iterable(self.ps for _ in range(count))
        return Images(list(ps), self.id)

    def sort(self):
        return Images(sorted(self.ps), self.id)

    def subsample(self, num_imgs) -> 'Images':
        assert len(self) >= num_imgs
        if len(self) == num_imgs:
            return self.copy()
        idxs = np.linspace(0, len(self) - 1, num_imgs, dtype=np.int)
        ps = [self.ps[idx] for idx in idxs]
        return Images(ps, f'{self.id}_s{num_imgs}')

    def filter(self, filter_fn, postfix=None):
        if not postfix:
            postfix = '_filtered'
        ps = list(filter(filter_fn, self.ps))
        return Images(ps, self.id + postfix)

    def copy_to(self, out_dir, verbose=False):
        os.makedirs(out_dir, exist_ok=True)
        for p in self.ps:
            out_p = os.path.join(out_dir, os.path.basename(p))
            if verbose:
                print(f'-> {out_p}')
            shutil.copy(p, out_p)

    def __str__(self):
        return repr(self)

    def copy(self):
        return Images(self.ps[:], self.id)


class InvalidFileError(Exception):
    pass


def _check_img(p, create_without_shitty=False):
    try:
        img = Image.open(p)
    except OSError as e:
        raise InvalidFileError(f'Cannot open file {p}: {e}')
    if create_without_shitty:
        smallest_size = min(img.size)
        if img.mode != 'RGB':
            return smallest_size, 'non_rgb'
        return smallest_size, None

    img = np.array(img)
    h, w = img.shape[:2]
    smallest_size = min(h, w)
    if len(img.shape) != 3 or img.shape[2] != 3:
        return smallest_size, 'non_rgb'
    ratio = min(h, w) / max(h, w)
    if ratio < 0.4:
        return smallest_size, 'ratio'
    top_pixels = img[:20, ...]
    if len(np.unique(top_pixels)) < 2:
        return smallest_size, 'border'
    bright_or_dark = np.sum(img < 10) + np.sum(img > 245)
    if bright_or_dark/np.prod(img.shape) > 0.5:
        return smallest_size, 'bright_or_dark'
    # hsv_discard = _hsv_discard(img)
    # if hsv_discard:
    #     return smallest_size, hsv_discard
    # all good
    return smallest_size, None


def _hsv_discard(im_rgb):
    im_hsv = skimage.color.rgb2hsv(im_rgb)
    mean_hsv = np.mean(im_hsv, axis=(0, 1))
    h, s, v = mean_hsv
    if s > 0.9:
        return f'hsv_s{s:.1f}'
    if v > 0.8:
        return f'hsv_v{v:.1f}'
    return None


class _ProcessHelper(object):
    def __init__(self, create_without_shitty):
        self.create_without_shitty = create_without_shitty

    def process(self, p):
        if os.path.getsize(p) == 0:
            print('*** BAD FILE', p)
            return None
        try:
            smallest_size, shitty = _check_img(p, self.create_without_shitty)
            return os.path.basename(p), smallest_size, shitty
        except ValueError as e:
            print('*** Caught', e, p)
            return None
        except InvalidFileError as e:
            print('*** Caught', e, p)
            return None


def iter_images(root_dir, num_folder_levels=0):
    fns = sorted(os.listdir(root_dir))
    for fn in fns:
        if num_folder_levels > 0:
            dir_p = os.path.join(root_dir, fn)
            if os.path.isdir(dir_p):
                print('Recursing into', fn)
                yield from iter_images(dir_p, num_folder_levels - 1)
            continue
        _, ext = os.path.splitext(fn)
        if ext.lower() in IMG_EXTENSIONS:
            yield os.path.join(root_dir, fn)


def _create_pickle(root_p, pkl_p, distributed_create, create_without_shitty, num_folder_levels=0):
    print(f'Globbing {root_p}...')
    ps = list(iter_images(root_p, num_folder_levels))
    print(f'Found {len(ps)} files!')
    if distributed_create:
        print('--distributed_create, filtering files...')
        ps = [p for _, p in task_array.job_enumerate(ps)]
        print(f'{len(ps)} files left!')
    if create_without_shitty:
        print('--create_without_shitty given')
    database = []
    start = time.time()
    shitty_reasons = defaultdict(int)
    num_processes = int(os.environ.get('MAX_PROCESS', 16)) if not distributed_create else 8
    h = _ProcessHelper(create_without_shitty)
    with multiprocessing.Pool(processes=num_processes) as pool:
        for i, res in enumerate(pool.imap_unordered(h.process, ps)):
            if res is None:
                continue
            database.append(res)
            _, _, shitty_reason = res
            if shitty_reason:
                shitty_reasons[shitty_reason] += 1
            if i > 0 and i % 100 == 0:
                time_per_img = (time.time() - start) / (i + 1)
                time_remaining = time_per_img * (len(ps) - i)
                info = f'\r{time_per_img:.2e} s/img | {i / len(ps) * 100:.1f}% | {time_remaining / 60:.1f} min remaining'
                if shitty_reasons and i % 1000 == 0:
                    info += ' ' + '|'.join(f'{reason}:{count}' for reason, count in
                                           sorted(shitty_reasons.items(), key=operator.itemgetter(1)))
                print(info, end='', flush=True)
    info = '|'.join(f'{reason}:{count}' for reason, count in
                    sorted(shitty_reasons.items(), key=operator.itemgetter(1)))
    print('\nshitty_reasons', info)
    print('Processed all')
    if len(database) == 0:
        raise ValueError(f'No images found in {root_p}!')
    print(f'\nGlobbed {len(database)} images, storing in {pkl_p}!')
    with open(pkl_p, 'wb') as fout:
        pickle.dump(database, fout)


def _distributed_suffix(task_id=task_array.TASK_ID):
    return f'_dist{task_id}'


def _get_pickle(p, reset=False, distributed_create=False, create_without_shitty=False, num_folder_levels=0):
    distributed_suffix = '' if not distributed_create else _distributed_suffix()
    pkl_p = os.path.join(p, PKL_NAME) + distributed_suffix
    lock_p = os.path.join(p, '.cached_glob.lock') + distributed_suffix
    print(f'Getting lock for {pkl_p}: {os.path.basename(lock_p)} [reset: {reset}]...')
    with fasteners.InterProcessLock(lock_p):
        if reset or not os.path.isfile(pkl_p):
            _create_pickle(p, pkl_p, distributed_create, create_without_shitty, num_folder_levels)
    return pkl_p


def _joined(images: Generator[Images, None, None]):
    joined_ps = []
    joined_id = []
    images_per_ids = {}
    for i in images:
        images_per_ids[i.id] = len(i.ps)
        joined_ps += i.ps
        joined_id.append(i.id)
    print('Joined:\n' + '\n'.join(f'{ds}: {c}' for ds, c in sorted(images_per_ids.items())))
    return Images(joined_ps, '&'.join(joined_id))


CachedImage = namedtuple('CachedImage', ['name', 'smallest_size', 'shitty', 'full_p'])


def cached_listdir_imgs_max(p, max_size=None, discard_shitty=True):
    ps = []
    filtered_max = 0
    with timer.execute(f'>>> filter [max_size={max_size}; discard_s={discard_shitty}]'):
        for img in _iter_imgs(p):
            if max_size and img.smallest_size >= max_size:
                filtered_max += 1
                continue
            if discard_shitty and img.shitty:
                continue
            ps.append(img.full_p)
    print('Filtered', filtered_max, 'imgs!')
    return Images(ps, id=f'{os.path.basename(p.rstrip(os.path.sep))}_{max_size}_dS={discard_shitty}')


def make_cache_fast(p):
    _get_pickle(p,
                reset=False, distributed_create=False,
                create_without_shitty=True, num_folder_levels=0)


def cached_listdir_imgs(p, min_size=None, discard_shitty=True) -> Images:
    if isinstance(p, list):
        return _joined(cached_listdir_imgs(p_, min_size, discard_shitty)
                       for p_ in p)
    if isinstance(p, tuple):
        p, resample = p
        if isinstance(resample, int):
            return cached_listdir_imgs(p, min_size, discard_shitty).repeat(resample)
        if isinstance(resample, float):
            assert 0 <= resample < 1, resample
            images = cached_listdir_imgs(p, min_size, discard_shitty)
            subsample = int(resample * len(images))
            return images.subsample(subsample)
        raise ValueError('Invalid type for resample:', resample)

    if not os.path.isdir(p):
        raise NotADirectoryError(p)

    ps = []
    with timer.execute(f'>>> filter [min_size={min_size}; discard_s={discard_shitty}]'):
        for img in _iter_imgs(p):
            if min_size and img.smallest_size < min_size:
                continue
            if discard_shitty and img.shitty:
                continue
            ps.append(img.full_p)
    return Images(ps, id=f'{os.path.basename(p.rstrip(os.path.sep))}_{min_size}_dS={discard_shitty}')


def _iter_imgs(p, reset=False, distributed_create=False, create_without_shitty=False, num_folder_levels=0):
    with open(_get_pickle(p, reset, distributed_create, create_without_shitty, num_folder_levels), 'rb') as f:
        imgs = pickle.load(f)

    for img_name, smallest_size, shitty in imgs:
        yield CachedImage(img_name, smallest_size, shitty,
                          full_p=os.path.join(p, img_name))


def _overwrite_pickle(p, filter_fn):
    pkl_p = _get_pickle(p)
    with open(pkl_p, 'rb') as f:
        imgs = pickle.load(f)

    len_imgs_before = len(imgs)
    imgs = [(img_name, smallest_size, shitty)
            for img_name, smallest_size, shitty in imgs
            if filter_fn(img_name)]
    len_imgs_after = len(imgs)
    if len_imgs_after == len_imgs_before:
        print('Nothing filtered!')
        return

    print(f'Before: {len_imgs_before} // After filter: {len_imgs_after}')
    with open(pkl_p, 'wb') as f:
        pickle.dump(imgs, f)


def _copy_to(img: CachedImage, copy_shitty_out_dir):
    sub_dir = os.path.join(copy_shitty_out_dir, img.shitty)
    os.makedirs(sub_dir, exist_ok=True)
    out_p = os.path.join(sub_dir, img.name)
    shutil.copy(img.full_p, out_p)


def _copy_non_shitty(images_it, out_dir, max_imgs):
    os.makedirs(out_dir, exist_ok=True)
    imgs = [img for img in images_it if not img.shitty]
    print(f'Found {len(imgs)} non-shitty images, copying {max_imgs}!')
    if max_imgs < len(imgs):
        idxs = np.linspace(0, len(imgs) - 1, max_imgs, dtype=np.int).tolist()
        imgs = [imgs[i] for i in idxs]
    for img in imgs:
        out_p = os.path.join(out_dir, img.name)
        shutil.copy(img.full_p, out_p)


def _wait_all_pkls_exist(img_dir):
    all_pkls = [os.path.join(img_dir, PKL_NAME) + _distributed_suffix(task_id)
                for task_id in range(task_array.NUM_TASKS)]
    print('Waiting for:', '/'.join(map(os.path.basename, all_pkls)))

    wait_time = 0
    while True:
        if all(map(os.path.isfile, all_pkls)):
            time.sleep(10)
            print('All exist!')
            return all_pkls
        time.sleep(1)
        wait_time += 1
        if wait_time % 15 == 0:
            print(f'\rWaiting for all pkls to exists. Waited for {wait_time}', end='', flush=True)


def _wait_and_merge(img_dir):
    all_pkls = _wait_all_pkls_exist(img_dir)
    final = []
    for p in all_pkls:
        with open(p, 'rb') as f:
            final += pickle.load(f)
    print(f'Found {len(final)} entries: {final[:10]}, writing...')
    final = sorted(final)
    with open(os.path.join(img_dir, PKL_NAME), 'wb') as fout:
        pickle.dump(final, fout)
    print('Removing other globs')
    for p in all_pkls:
        os.remove(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('img_dir')
    p.add_argument('--remove_empty', action='store_true')
    p.add_argument('--reset', action='store_true')
    p.add_argument('--distributed_create', action='store_true')
    p.add_argument('--filter_min_size', '-fm', type=int)
    p.add_argument('--num_folder_levels', '-r', type=int, default=0)
    p.add_argument('--filter_shitty', '-fs', action='store_true')
    p.add_argument('--check_exists', action='store_true')
    p.add_argument('--create_without_shitty', action='store_true')
    p.add_argument('--copy_shitty', '-cp', type=str, metavar='OUT_DIR', help=f'May use {IMG_DIR_REPL} as placeholder')
    p.add_argument('--copy_non_shitty', '-cpn', type=str, metavar='OUT_DIR,COUNT', help=f'May use {IMG_DIR_REPL} as placeholder')

    flags = p.parse_args()

    if flags.remove_empty:
        print('--remove_empty')
        empty_files = {os.path.basename(cached_img.full_p) for cached_img in _iter_imgs(flags.img_dir)
                       if os.path.getsize(cached_img.full_p) == 0}
        if empty_files:
            print(f'Found {len(empty_files)} empty files! {empty_files}')
            _overwrite_pickle(flags.img_dir, filter_fn=lambda img_name_: img_name_ not in empty_files)
            for empty_file_name in empty_files:
                p = os.path.join(flags.img_dir, empty_file_name)
                assert os.path.isfile(p)
                print(f'Deleting {p}...')
                os.remove(p)
        return

    if flags.distributed_create:
        assert task_array.NUM_TASKS > 1

    def _replace_img_dir_repl(p_):
        if IMG_DIR_REPL in p_:
            return p_.replace(IMG_DIR_REPL, flags.img_dir.rstrip(os.path.sep))
        return p_

    if flags.copy_non_shitty:
        assert ',' in flags.copy_non_shitty
        out_dir, count = flags.copy_non_shitty.split(',')
        out_dir = _replace_img_dir_repl(out_dir)
        _copy_non_shitty(_iter_imgs(flags.img_dir, flags.reset), out_dir, int(count))
        return

    copy_shitty_out_dir = flags.copy_shitty
    if copy_shitty_out_dir:
        copy_shitty_out_dir = _replace_img_dir_repl(copy_shitty_out_dir)
        if os.path.isdir(copy_shitty_out_dir):
            raise FileExistsError(copy_shitty_out_dir)

        flags.filter_shitty = True
        os.makedirs(copy_shitty_out_dir, exist_ok=True)

    counter = defaultdict(int)
    for img in _iter_imgs(flags.img_dir,
                          flags.reset, flags.distributed_create, flags.create_without_shitty,
                          flags.num_folder_levels):
        if flags.check_exists:
            assert os.path.isfile(img.full_p)
        counter['total'] += 1
        if flags.filter_min_size and img.smallest_size < flags.filter_min_size:
            print(f'Too small: {img.name} {img.smallest_size}')
            counter['too_small'] += 1
        if flags.filter_shitty and img.shitty:
            print(f'Shitty: {img.name} {img.shitty}')
            counter[img.shitty] += 1
            if copy_shitty_out_dir:
                _copy_to(img, copy_shitty_out_dir)
    print(', '.join(f'{key}: {count}' for key, count in counter.items()))

    if flags.distributed_create and task_array.TASK_ID == 1:
        print('TASK_ID==1, waiting to merge')
        _wait_and_merge(flags.img_dir)

    print('\nSUCCESS')



if __name__ == '__main__':
    main()
