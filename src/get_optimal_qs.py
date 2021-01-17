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
import csv

import numpy as np
import pytorch_ext as pe
import torch

from PIL import Image
import os
import argparse

from fjcommon import timer

from blueprints.classifier_blueprint import load_classifier
from dataloaders.cached_listdir_imgs import cached_listdir_imgs
from helpers.quantized_tensor import NormalizedTensor, SymbolTensor

OPTIMAL_QS_TXT = 'optimal_q.txt'


def read(p):
    if not p.endswith(OPTIMAL_QS_TXT):
        p = os.path.join(p, OPTIMAL_QS_TXT)
    if not os.path.isfile(p):
        print('No file', p)
        return None
    cache = {}
    with open(p, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip
        for fn, q in reader:
            cache[fn] = int(q)
    return cache


def _get(img_dir, clf_ckpt_p):
    out_file = os.path.join(img_dir, OPTIMAL_QS_TXT)

    clf = load_classifier(clf_ckpt_p)

    t = timer.TimeAccumulator()
    opt = {}
    for i, p in enumerate(cached_listdir_imgs(img_dir, discard_shitty=False).ps):
        with t.execute():
            img = torch.from_numpy(np.array(Image.open(p))).to(pe.DEVICE, non_blocking=True).permute(2, 0, 1)
            assert img.shape[0] == 3
            img = img.unsqueeze(0)
            img = SymbolTensor(img.long(), L=256).to_norm().get()
            q = clf.get_q(img)
            opt[os.path.splitext(os.path.basename(p))[0]] = q
        if i > 0 and i % 10 == 0:
            print(i, t.mean_time_spent())
    with open(out_file, 'w', newline='') as csvfile:
        w = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        w.writerow(['fn', 'q'])
        for filename, q in sorted(opt.items()):
            w.writerow([filename, q])
    print('Created', out_file)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('img_dir')
    p.add_argument('clf_ckpt_p')
    flags = p.parse_args()
    _get(flags.img_dir, flags.clf_ckpt_p)


if __name__ == '__main__':
    main()