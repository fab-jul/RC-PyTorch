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
import time

import numpy as np
from PIL import Image, ImageFile
from fjcommon import timer

import get_optimal_qs
import task_array
from dataloaders import cached_listdir_imgs
from lossy import other_codecs

BALLE_MODEL = 'bmshj2018-factorized-mse-'


def open_safe(p):
    try:
        return Image.open(p)
    except ValueError as e:
        print(f'*** Caught {e}, {p}, saving again')
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.open(p).save(p, optimize=True)
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        try:
            return Image.open(p)
        except ValueError as e:
            raise ValueError(f'Cannot catch: {e} for {p}')


def _balle_compress(inp_p, out_p, params=None):
    assert '{}' in out_p
    # OUTPUT_{}.png --> OUTPUT___tmp__.tfci
    out_balle_tmp = os.path.splitext(out_p.replace('{}', '__tmp__'))[0] + '.tfci'
    other_codecs.balle_compress(BALLE_MODEL, inp_p, os.path.abspath(out_balle_tmp), q='8')
    # set actual outputfilename
    bpp = other_codecs.bpp_of_balle_image(inp_p, out_balle_tmp)
    # OUTPUT_{}.png --> OUTPUT_3.11234.png
    out_p = out_p.replace('{}', f'{bpp:.5f}')
    other_codecs.decode_balle_to_png(out_balle_tmp, out_p)
    assert os.path.isfile(out_p)
    return out_p



def iter_q_in_flag(flag):
    qa, qb = map(int, flag.lstrip('A').split('_'))
    for q in range(qa, qb + 1):
        yield q


def _bpg_compress(inp_p, out_p_template, params):
    tmp_dir = params['tmp_dir']
    q = params.get('q', '1')
    try:
        q = int(q)
    except ValueError:
        raise ValueError('Q should be convertible to int:' + str(q))
    bpg_out = other_codecs.bpg_compress(inp_p, q=q, tmp_dir=tmp_dir)
    img = open_safe(inp_p)
    num_pixels = np.prod(img.size)
    bpp = os.path.getsize(bpg_out) * 8 / num_pixels
    out_p = out_p_template.replace('{}', f'{bpp:.5f}')
    other_codecs.decode_bpg_to_png(bpg_out, out_p=out_p)
    os.remove(bpg_out)
    return out_p


COMPRESSORS = {
    'balle': _balle_compress,
    'bpg': _bpg_compress
}


def _open_img_as_np(p, dtype=np.uint8):
    return np.array(Image.open(p)).astype(dtype)


def _save_diff(img_p, out_p, out_dir):
    filename = os.path.basename(img_p)
    filename_no_ext, _ = os.path.splitext(filename)
    diff_p = os.path.join(out_dir, filename_no_ext + '.npy')
    print(f'*** Saving diff {filename} in {out_dir}')
    diff = _open_img_as_np(img_p, np.int16) - _open_img_as_np(out_p, np.int16)
    print(f'Diff Range: {diff.min()}, {diff.max()}')
    np.save(diff_p, diff)


class Compressor(object):
    def __init__(self, compress_fn, outdir_base, save_diff, force, params=None):
        self.compress_fn = compress_fn
        self.save_diff = save_diff
        self.force = force
        self.outdir_base = outdir_base

        self.params = params
        self.optimal_qs = None

        # TODO
        self.files_that_exist = {}

        for outdir, _ in self._unroll_params(params):
            print('Creating', outdir)
            os.makedirs(outdir, exist_ok=True)


    def _unroll_params(self, params: dict, filename=None):
        if 'q' in params:
            q = params['q']
            if isinstance(q, str) and q.startswith('A'):  # A12_14 -> all between 12 and 14 inclusive
                for q in iter_q_in_flag(q):
                    params_q = params.copy()
                    params_q['q'] = q
                    yield self.get_outdir_for_q(q), params_q
            elif isinstance(q, str) and q.startswith('R'):
                qa, qb = map(int, q.lstrip('R').split('_'))
                params_q = params.copy()
                params_q['q'] = random.randint(qa, qb)
                yield self.get_outdir_for_q(q), params_q  # just use R12_14 as outdir!!
            elif isinstance(q, str) and q.startswith('OPT'):
                offset = q.replace('OPT', '')
                if offset:
                    if offset.startswith('P'):
                        offset = int(offset.replace('P', ''))
                    elif offset.startswith('M'):
                        offset = -int(offset.replace('M', ''))
                    else:
                        raise ValueError(f'Invalid offset: {offset}')
                    if offset:
                        print(offset)
                else:
                    offset = 0
                params_q = params.copy()
                if filename:
                    assert self.optimal_qs
                    params_q['q'] = self.optimal_qs[filename] + offset
                yield self.get_outdir_for_q(q), params_q
            else:
                yield self.get_outdir_for_q(params['q']), params
        else:
            raise ValueError(params)

    @staticmethod
    def bpp_from_compressed_file(p):
        fn = os.path.basename(p)
        assert 'bpp' in fn, p
        fn, ext = os.path.splitext(fn)
        assert ext == '.png', p
        bpp = fn.split('bpp')[1]
        return float(bpp)

    @staticmethod
    def filename_without_bpp(p):
        fn = os.path.basename(p)
        assert 'bpp' in p, p
        return fn.split('bpp')[0]

    def get_outdir_for_q(self, q):
        # TODO: only works with BPG!!
        return self.outdir_base + f'_bpg_q{q}'

    def compress(self, img_p):
        filename = os.path.basename(img_p)
        filename_no_ext, _ = os.path.splitext(filename)

        for outdir, params in self._unroll_params(self.params, filename_no_ext):

            # if not self.force and (filename_no_ext in self.files_that_exist):
            #     print(f'{filename_no_ext} already exists, skipping...')
            #     out_fn = self.files_that_exist[filename_no_ext]
            #     out_p = os.path.join(outdir, out_fn)
            # else:

            # it is expected that self.compress_fn saves bpp into the filename
            out_p_template = os.path.join(outdir, filename_no_ext + 'bpp{}.png')
            # print(f'*** Compression {filename}...')
            # with timer.execute('>> compress [s]'):
            out_p = self.compress_fn(img_p, out_p_template, params)

            if self.save_diff:
                print(f'*** Saving diff {filename}...')
                _save_diff(img_p, out_p, out_dir=self.save_diff)


def compress(compressor: Compressor, indir, discard_shitty):
    ds = cached_listdir_imgs.cached_listdir_imgs(indir, min_size=None, discard_shitty=discard_shitty)
    imgs = ds.ps
    compressor.optimal_qs = get_optimal_qs.read(indir)
    if compressor.optimal_qs:
        print('Optimal Qs:', len(compressor.optimal_qs))
    assert len(imgs) > 0, f'no matches for {indir}'
    num_imgs_to_process = len(imgs) // task_array.NUM_TASKS

    images_of_job = [p for _, p in task_array.job_enumerate(imgs)]
    N_orig = len(images_of_job)
    images_of_job = [p for p in images_of_job if os.path.splitext(os.path.basename(p))[0]
                     not in compressor.files_that_exist]
    N = len(images_of_job)

    start = time.time()
    num_process = 2 if task_array.NUM_TASKS > 1 else int(os.environ.get('MAX_PROCESS', 16))
    print(f'{task_array.JOB_ID}:',
          f'Compressing {N}/{N_orig} images ({ds.id}) using {num_process} processes',
          f'in {task_array.NUM_TASKS} tasks...')

    with multiprocessing.Pool(processes=num_process) as pool:
        for i, _ in enumerate(pool.imap_unordered(compressor.compress, images_of_job)):
            if i > 0 and i % 5 == 0:
                time_per_img = (time.time() - start) / (i + 1)
                time_remaining = time_per_img * (N - i)
                print(f'\r{time_per_img*num_process:.2e} s/img | '
                      f'{i / N * 100:.1f}% | {time_remaining / 60:.1f} min remaining',
                      end='', flush=True)


def _parse_params(params: str, target: dict):
    if not target:
        target = {}
    for p in params.split(','):
        key, value = p.split('=')
        print(f'Setting param {key}={value}')
        target[key] = value
    return target


def main():
    p = argparse.ArgumentParser()
    p.add_argument('codec', choices=COMPRESSORS.keys())
    p.add_argument('indir')
    p.add_argument('outdir_base')
    p.add_argument('--discard_shitty', type=int, required=True)

    p.add_argument('--save_diff', type=str, metavar='SAVE_DIFF_DIR')
    p.add_argument('--force', '-f', action='store_true')
    p.add_argument('--params', type=str)
    flags = p.parse_args()

    if flags.save_diff:
        os.makedirs(flags.save_diff, exist_ok=True)

    if flags.codec == 'bpg':
        tmp_dir = flags.outdir_base.rstrip(os.sep) + '_bpg_out_tmp'
        os.makedirs(tmp_dir, exist_ok=True)
        params = {
            'tmp_dir': tmp_dir
        }
    else:
        params = None

    if flags.params:
        params = _parse_params(flags.params, target=params)

    c = Compressor(COMPRESSORS[flags.codec], flags.outdir_base, flags.save_diff, flags.force, params)
    compress(c, flags.indir, flags.discard_shitty != 0)
    print(f'DONE Compressed all')



if __name__ == '__main__':
    main()
