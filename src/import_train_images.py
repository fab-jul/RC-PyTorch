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
import multiprocessing
import os
import random
import shutil
import time
import warnings
from os.path import join

import PIL
import numpy as np
import skimage.color
from PIL import Image

from dataloaders.cached_listdir_imgs import iter_images, cached_listdir_imgs

# task_array is not released. It's used by us to batch process on our servers. Feel free to replace with whatever you
# use. Make sure to set NUM_TASKS (number of concurrent processes) and set job_enumerate to a function that takes an
# iterable and only yield elements to be processed by the current process.
try:
    from task_array import NUM_TASKS, job_enumerate
except ImportError:
    NUM_TASKS = 1
    job_enumerate = enumerate

warnings.filterwarnings("ignore")


QUALITY = 95

DOWNSAMPLING = {
    'bicubic': PIL.Image.BICUBIC,
    'lanczos': PIL.Image.LANCZOS
}


get_fn = lambda p_: os.path.splitext(os.path.basename(p_))[0]


class Helper(object):
    def __init__(self, out_dir_clean, out_dir_discard, res: int,
                 crop4, crop16, random_scale,
                 downsampling, downdown,
                 prepend_dir):
        print(f'Creating {out_dir_clean}, {out_dir_discard}...')
        os.makedirs(out_dir_clean, exist_ok=True)
        os.makedirs(out_dir_discard, exist_ok=True)
        self.out_dir_clean = out_dir_clean
        self.out_dir_discard = out_dir_discard

        print('Getting processed images...')
        self.images_cleaned = set(map(get_fn, os.listdir(out_dir_clean)))
        self.images_discarded = set(map(get_fn, os.listdir(out_dir_discard)))
        print(f'Found {len(self.images_cleaned) + len(self.images_discarded)} processed images.')

        self.res = res

        self.crop4 = crop4
        self.crop16 = crop16
        self.random_scale = random_scale
        self.downdown = downdown
        self.downsampling = downsampling
        self.prepend_dir = prepend_dir

    def process_all_in(self, input_dir, filter_imgs_dir):
        images_dl = iter_images(input_dir)  # generator of paths

        # files this job should comperss
        files_of_job = [p for _, p in job_enumerate(images_dl)]
        # files that were compressed already by somebody (i.e. this job earlier)
        processed_already = self.images_cleaned | self.images_discarded
        # resulting files to be compressed
        files_of_job = [p for p in files_of_job if get_fn(p) not in processed_already]

        if filter_imgs_dir:
            ps_orig = cached_listdir_imgs(filter_imgs_dir, discard_shitty=True).ps
            fns_to_use = set(map(get_fn, ps_orig))
            print('Filtering with', len(fns_to_use), 'filenames. Before:', len(files_of_job))
            files_of_job = [p for p in files_of_job if get_fn(p) in fns_to_use]
            print('Filtered, now', len(files_of_job))

        N = len(files_of_job)
        if N == 0:
            print('Everything processed / nothing to process.')
            return

        num_process = 2 if NUM_TASKS > 1 else int(os.environ.get('MAX_PROCESS', 16))
        print(f'Processing {N} images using {num_process} processes in {NUM_TASKS} tasks...')

        start = time.time()
        predicted_time = None
        with multiprocessing.Pool(processes=num_process) as pool:
            for i, clean in enumerate(pool.imap_unordered(self.process, files_of_job)):
                if i > 0 and i % 100 == 0:
                    time_per_img = (time.time() - start) / (i + 1)
                    time_remaining = time_per_img * (N - i)
                    if not predicted_time:
                        predicted_time = time_remaining
                    print(f'\r{time_per_img:.2e} s/img | {i / N * 100:.1f}% | {time_remaining / 60:.1f} min remaining', end='', flush=True)
        if predicted_time:
            print(f'Actual time: {(time.time() - start) / 60:.1f} // predicted {predicted_time / 60:.1f}')

    def process(self, p_in):
        fn, ext = os.path.splitext(os.path.basename(p_in))
        if fn in self.images_cleaned:
            return 1
        if fn in self.images_discarded:
            return 0
        try:
            im = Image.open(p_in)
            if self.crop4 or self.crop16:
                # if should_discard(im):
                #     return 0
                _crop_fn = _crop4 if self.crop4 else _crop16
                for i, im_crop in enumerate(_crop_fn(im)):
                    p_out = join(self.out_dir_clean, f'{fn}_{i}.png')
                    print(p_out)
                    im_crop.save(p_out)
                return 1
            if self.downdown:
                if self.prepend_dir:
                    prepend = p_in.split(os.path.sep)[-2]
                    fn = prepend + '_' + fn
                p_out = join(self.out_dir_clean, f'{fn}.png')
                im.save(p_out)
                for fac in [0.9, 0.8, 0.7, 0.6, 0.5]:
                    im_scale = rescale(im, fac)
                    p_out = join(self.out_dir_clean, f'{fn}_{fac:.1f}.png')
                    im_scale.save(p_out)
                return 1
            if self.random_scale:
                im_out = random_resize(im, min_res=self.random_scale)
                if im_out is None:
                    return 0
                im_out.save(join(self.out_dir_clean, fn + '.png'))
                return 1
            # old code ----
            im2 = resize_or_discard(im, self.res, downsampling=self.downsampling)
            if im2 is not None:
                im2.save(join(self.out_dir_clean, fn + '.png'))
                return 1
            else:
                p_out = join(self.out_dir_discard, os.path.basename(p_in))
                shutil.copy(p_in, p_out)
                return 0
        except OSError as e:
            print(e)
            return 0


def _crop16(im):
    for im_cropped in _crop4(im):
        yield from _crop4(im_cropped)


def _crop4(im):
    w, h = im.size
    #               (left, upper, right, lower)
    imgs = [im.crop((0, 0, w//2, h//2)),  # top left
            im.crop((0, h//2, w//2, h)),  # bottom left
            im.crop((w//2, 0, w, h//2)),  # top right
            im.crop((w//2, h//2, w, h)),  # bottom right
            ]

    assert sum(np.prod(i.size) for i in imgs) == np.prod(im.size)
    return imgs


def test_this_thit():
    i = Image.open('/Users/fabian/Documents/PhD/data/fixedimg.jpg')
    _crop4(i)


def resize_or_discard(im, res: int, verbose=False, downsampling=PIL.Image.BICUBIC):
    im2 = resize(im, res, downsampling)
    if im2 is None:
        return None
    if should_discard(im2):
        return None
    return im2


def rescale(im, scale):
    W, H = im.size

    W2 = round(W * scale)
    H2 = round(H * scale)

    try:
        # TODO
        return im.resize((W2, H2), resample=Image.LANCZOS)
    except OSError as e:
        print('*** im.resize error', e)
        return None



def resize(im, res, downsampling=PIL.Image.BICUBIC):
    """ scale longer side to `res`. """
    W, H = im.size
    D = max(W, H)
    scaling_factor = float(res) / D
    # image is already the target resolution, so no downscaling possible...
    if scaling_factor > 0.95:
        return None
    W2 = round(W * scaling_factor)
    H2 = round(H * scaling_factor)
    try:
        # TODO
        return im.resize((W2, H2), resample=downsampling)
    except OSError as e:
        print('*** im.resize error', e)
        return None


MAX_SCALE = 0.8


def random_resize(im, min_res):
    """Scale longer side to `min_res`, but only if that scales by < MAX_SCALE."""
    W, H = im.size
    D = min(W, H)
    scale_min = min_res / D
    # Image is too small to downscale by a factor smaller MAX_SCALE.
    if scale_min > MAX_SCALE:
        return None

    # Get a random scale.
    scale = random.uniform(scale_min, MAX_SCALE)

    new_size = round(W * scale), round(H * scale)
    try:
        # Important: Use LANCZOS!
        return im.resize(new_size, resample=PIL.Image.LANCZOS)
    except OSError as e:  # Happens for corrupted images
        print('*** Caught im.resize error', e)
    return None


def should_discard(im):
    # modes found in train_0:
    # Counter({'RGB': 152326, 'L': 4149, 'CMYK': 66})
    if im.mode != 'RGB':
        return True

    im_rgb = np.array(im)
    im_hsv = skimage.color.rgb2hsv(im_rgb)
    mean_hsv = np.mean(im_hsv, axis=(0, 1))
    _, s, v = mean_hsv
    if s > 0.9:
        return True
    if v > 0.8:
        return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('base_dir')
    p.add_argument('dirs', nargs='*')
    p.add_argument('--out_dir_clean', required=True)
    p.add_argument('--out_dir_discard', required=True)
    p.add_argument('--resolution', '-r', type=int, default=768)
    p.add_argument('--crop4', action='store_true')
    p.add_argument('--crop16', action='store_true')
    p.add_argument('--random_scale', type=int)
    p.add_argument('--downdown', action='store_true')
    p.add_argument('--filter_with_dir', type=str)
    p.add_argument('--prepend_dir', action='store_true')
    p.add_argument('--downsampling', type=str, choices=DOWNSAMPLING.keys(), default='bicubic')

    flags = p.parse_args()
    h = Helper(flags.out_dir_clean, flags.out_dir_discard, flags.resolution,
               flags.crop4, flags.crop16, flags.random_scale,
               DOWNSAMPLING[flags.downsampling], flags.downdown, flags.prepend_dir)

    if not flags.dirs:
        flags.dirs = [os.path.basename(flags.base_dir)]
        flags.base_dir = os.path.dirname(flags.base_dir)

    for d in flags.dirs:
        h.process_all_in(join(flags.base_dir, d), flags.filter_with_dir)

    # for qsub logs
    print('\n\nDONE')


if __name__ == '__main__':
    main()
