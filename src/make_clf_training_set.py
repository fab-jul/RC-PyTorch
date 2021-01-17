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
import shutil
import sys
import os
import argparse

from dataloaders.cached_listdir_imgs import cached_listdir_imgs, make_cache_fast


def make_clf_training_set(training_set_dir):
    V = cached_listdir_imgs(training_set_dir,
                            min_size=512,
                            discard_shitty=False)
    clf_training_set_filenames = get_clf_training_set_filenames()

    # Make sure we have them all
    ps = set(map(os.path.basename, V.ps))
    missing = set(clf_training_set_filenames) - ps
    if missing:
        print(f'ERROR: Not all files found, missing {missing}!')
        sys.exit(1)

    # Create the subset folder
    out_dir = training_set_dir.rstrip(os.path.sep) + '_subset_clf'
    print(f'Creating {out_dir}...')
    os.makedirs(out_dir, exist_ok=True)

    print_every = max(len(clf_training_set_filenames) // 20, 1)  # Update every 5%
    for i, filename in enumerate(clf_training_set_filenames):
        if i > 0 and i % print_every == 0:
            percent = i / len(clf_training_set_filenames) * 100
            print(f'Update: {percent:.1f}% copied')
        in_p = os.path.join(training_set_dir, filename)
        out_p = os.path.join(out_dir, filename)
        if not os.path.isfile(out_p):
            shutil.copy(in_p, out_p)

    print('Caching files...')
    make_cache_fast(out_dir)

    print(f'\nSubfolder created at {out_dir}. Now run:\n'
          f'bash prep_bpg_ds.sh A11_17 {out_dir}')


def get_clf_training_set_filenames() -> list:
    this_file_p = os.path.dirname(os.path.abspath(__file__))
    clf_training_set_filenames_p = os.path.join(
        this_file_p, 'data', 'clf_training_set_filenames.txt')
    with open(clf_training_set_filenames_p) as f:
        return sorted(filter(None, f.read().split('\n')))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('training_set_dir')
    flags = p.parse_args()
    make_clf_training_set(flags.training_set_dir)


if __name__ == '__main__':
    main()