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
import glob
import os
import numpy as np
from collections import defaultdict

import torch
from PIL import Image

from vis.image_summaries import to_image


def stitch(images):
    """
    only really tested for 4 images, which are stiched as

        1   3
        2   4
    """
    assert len(images) == 4
    assert len(images[0].shape) == 4, images[0].shape

    if isinstance(images[0], np.ndarray):
        img_12 = np.concatenate((images[0], images[1]), axis=2)
        img_34 = np.concatenate((images[2], images[3]), axis=2)
        img = np.concatenate((img_12, img_34), axis=3)
        return img
    else:
        img_12 = torch.cat((images[0], images[1]), dim=2)
        img_34 = torch.cat((images[2], images[3]), dim=2)
        img = torch.cat((img_12, img_34), dim=3)
        return img



class ImageSaver(object):
    def __init__(self, out_dir, merge=False, trim=False):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.saved_fs = []
        if merge:
            self.merge = merge
            self.merger_cache = defaultdict(list)
        else:
            self.merge = False
        self.trim = trim

    def __str__(self):
        return 'ImageSaver({})'.format(self.out_dir)

    def _unpack(self, x, filename):
        prefix, postfix = filename
        if self.merge:
            # extract number
            fn, num = prefix[:-1], prefix[-1]
            # idx, fn = fn.split('_', 1)  # Added by multiscale_tester I think
            # check it's in assumed range
            assert 0 <= int(num) < self.merge, (num, prefix, self.merge)
            cache_key = fn + postfix
            cache = self.merger_cache[cache_key]
            print(f'Cache for {cache_key}: {len(cache)}')
            cache.append(x)
            if len(cache) == self.merge:  # got all crops!
                x = stitch(cache)  # torch or numpy!
                del self.merger_cache[cache_key]
                out_fn = fn + postfix
                print(f'Got {len(cache)} crops, saving as {out_fn}...')
                return x, out_fn
            return None, None  # nothing saved!
        else:
            filename = prefix + postfix
            return x, filename

    def _save(self, saver, x, filename, convert):
        # if self.trim:
        #     t = self.trim
        #     x = x[..., t:-t, t:-t]
        if isinstance(filename, tuple):
            x, filename = self._unpack(x, filename)
            if x is None:
                return None, None
        if convert:
            x = to_image(x.type(torch.uint8))
        print('*** Saving', filename)
        out_p = self.get_save_p(filename)
        saver(x, out_p)
        return x, filename

    def save_img(self, img, filename, convert_to_image=True):
        """
        :param img: image tensor, in {0, ..., 255}
        :param filename: output filename
        :param convert_to_image: if True, call to_image on img, otherwise assume this has already been done.
        :return:
        """
        return self._save(lambda x, p: Image.fromarray(x).save(p),
                          img, filename, convert_to_image)
        # if self.trim:
        #     t = self.trim
        #     img = img[..., t:-t, t:-t]
        # if isinstance(filename, tuple):
        #     img, filename = self._unpack(img, filename)
        #     if img is None:
        #         return None
        # if convert_to_image:
        #     img = to_image(img.type(torch.uint8))
        # out_p = self.get_save_p(filename)
        # return out_p

    def save_res(self, res, filename):
        assert isinstance(res, np.ndarray) and res.dtype == np.int16, (type(res), res.dtype)
        return self._save(lambda x, p: np.save(p, x),
                          res, filename, convert=False)
        # if self.trim:
        #     t = self.trim
        #     res = res[..., t:-t, t:-t]
        # if isinstance(filename, tuple):
        #     res, filename = self._unpack(res, filename)
        #     if res is None:
        #         return None, None
        # out_p = self.get_save_p(filename)
        #
        # return res, out_p

    def get_save_p(self, file_name):
        out_p = os.path.join(self.out_dir, file_name)
        self.saved_fs.append(file_name)
        return out_p

    def file_starting_with_exists(self, prefix):
        check_p = os.path.join(self.out_dir, prefix) + '*'
        print(check_p, glob.glob(check_p))
        return len(glob.glob(check_p)) > 0