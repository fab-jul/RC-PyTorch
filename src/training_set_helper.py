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
import urllib.request
import glob
import subprocess

from fjcommon import iterable_ext
import multiprocessing
import os
import sys

from dataloaders import cached_listdir_imgs


BASE_URL = "http://data.vision.ee.ethz.ch/mentzerf/rc_data/"
TAR_URLS = [
    BASE_URL + f'train_oi_r.tar.gz.{i}' for i in range(10)
]
TAR_GLOB = 'train_oi_r.tar.gz.*'


def download_and_unpack(outdir):
    os.makedirs(outdir, exist_ok=True)

    def reporthook(num_blocks_transferred, block_size, total_size):
        transferred_bytes = num_blocks_transferred * block_size
        percent = transferred_bytes / total_size * 100
        if int(percent * 100) % 10 == 0:
            print(f'\r-> {local_p} | {percent:.1f}% ...', end='', flush=True)

    for i, url in enumerate(TAR_URLS, 1):
        local_p = os.path.join(outdir, os.path.basename(url))
        if os.path.isfile(local_p):
            print(f'Exists, skipping: {url}')
            continue
        print(f'Downloading {url} [Part {i}/{len(TAR_URLS)}]...')
        urllib.request.urlretrieve(url, local_p, reporthook)
        print()  # To get a newline after the CRs above.

    tar_glob = os.path.join(outdir, TAR_GLOB)
    unpack(tar_glob)


def unpack(tar_glob):
    ps = glob.glob(tar_glob)
    if not ps:
        raise ValueError(tar_glob)

    print(f'Unpacking {len(ps)} tar files...')
    with multiprocessing.Pool(processes=len(ps)) as pool:
        completed = 0
        for out_p in pool.imap_unordered(_unpack_tar, enumerate(ps)):
            completed += 1
            print(f'Done: {out_p} ({completed}/{len(ps)} done)')


def _unpack_tar(idx_and_p):
    idx, p = idx_and_p
    for update in iter_progress_of_command(cmd=['tar', 'xvf', p]):
        print(f'JOB {idx}: {update}')


def create(indir, num_tars):
    ds = cached_listdir_imgs.cached_listdir_imgs(
          indir, min_size=None, discard_shitty=False)
    paths = ds.ps
    paths_per_tar = iterable_ext.chunks(paths, num_chunks=num_tars)

    print(f'Packing {len(paths)} images into {num_tars} tar files...')
    with multiprocessing.Pool(processes=num_tars) as pool:
        completed = 0
        for out_p in pool.imap_unordered(_pack_as_tar, enumerate(paths_per_tar)):
            completed += 1
            print(f'Done: {out_p} ({completed}/{num_tars} done)')


def _pack_as_tar(idx_and_ps):
    idx, ps = idx_and_ps
    if not ps:
        return
    # /some/path/train_dir/image.png
    some_p = ps[0]
    # /some/path/train_dir
    root = os.path.dirname(some_p)
    # /some/path
    wd = os.path.dirname(root)
    out_p = root.rstrip(os.path.sep) + f'.tar.gz.{idx}'
    if idx == 0:
        print(f'Files will be at {out_p[:-1]}*')
    # ditch the wd prefix
    normalized_ps = [p[len(wd):].lstrip(os.path.sep) for p in ps]
    for percent in iter_progress_of_command(
          cmd=['tar', 'czvf', out_p] + normalized_ps,
          total_expected_lines=len(ps),
          wd=wd):
        print(f'JOB {idx}: {percent * 100:.1f}%')
    return out_p


def iter_progress_of_command(cmd, total_expected_lines=None, wd=None):
    proc = subprocess.Popen(cmd, cwd=wd,
                            stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE)
    if total_expected_lines:
        print_every = max(total_expected_lines // 50, 1)
    else:
        print_every = 100

    completed = 0
    for _ in proc.stdout:
        completed += 1
        if completed % print_every == 0:
            if total_expected_lines:
                percent = completed / total_expected_lines
                yield f'{percent * 100:.1f}%'
            else:
                yield f'{completed} done'


def main():
    p = argparse.ArgumentParser()
    mode_parsers = p.add_subparsers(dest='mode')
    create_p = mode_parsers.add_parser('create')
    create_p.add_argument('indir')
    create_p.add_argument('--num_chunks', default=10, type=int)
    unpack_p = mode_parsers.add_parser('unpack')
    unpack_p.add_argument('tar_glob')
    unpack_p = mode_parsers.add_parser('download_and_unpack')
    unpack_p.add_argument('outdir')

    flags = p.parse_args()
    if flags.mode == 'create':
        create(flags.indir, flags.num_chunks)
    elif flags.mode == 'unpack':
        unpack(flags.tar_glob)
    elif flags.mode == 'download_and_unpack':
        download_and_unpack(flags.outdir)
    else:
        p.print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
