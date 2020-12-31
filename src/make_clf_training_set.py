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
import argparse

from dataloaders.cached_listdir_imgs import cached_listdir_imgs


def make_clf_training_set(training_set_dir):
    V = cached_listdir_imgs(training_set_dir,
                            min_size=512,
                            discard_shitty=False)
    print(len(V))
    clf_training_set_filenames = get_clf_training_set_filenames()
    ps = set(map(os.path.basename, V.ps))
    print(set(clf_training_set_filenames) - ps)


def get_clf_training_set_filenames() -> list:
    this_file_p = os.path.dirname(os.path.abspath(__file__))
    clf_training_set_filenames_p = os.path.join(
        this_file_p, 'data', 'clf_training_set_filenames.txt')
    with open(clf_training_set_filenames_p) as f:
        return f.read().split('\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('training_set_dir')
    flags = p.parse_args()
    make_clf_training_set(flags.training_set_dir)


if __name__ == '__main__':
    main()