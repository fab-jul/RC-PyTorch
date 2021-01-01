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
import collections
import csv
import os
import pickle
import re
from collections import namedtuple, defaultdict
from contextlib import contextmanager
from typing import Dict

import fasteners
import numpy as np
import torch

from blueprints.enhancement_blueprint import enhancement_loss_lt, \
    EnhancementLoss, QStrategy, EnhancementBlueprint
from helpers.quantized_tensor import SymbolTensor

CACHE_PKL = 'cache.pkl'


# Uniquely identifies a test run of some experiment
# dataset_id comes from Testset.id, which is 'FOLDERNAME_NUMIMGS'

TestID = namedtuple('TestID', ['dataset_id', 'restore_itr'])


def get_test_log_dir_root(log_dir):
    return log_dir.rstrip(os.path.sep) + '_test'


class TestResults(object):
    def __init__(self):
        # dict mapping metric -> img -> result
        self.per_img_results = defaultdict(dict)

    def set(self, filename, metric, result):
        try:
            self.per_img_results[metric][filename] = result.item()
        except AttributeError:
            self.per_img_results[metric][filename] = result

    def set_from_loss(self, loss_out, filename):
        if isinstance(loss_out, EnhancementLoss):
            self.set(filename, 'bpsp', loss_out.bpsp_base + loss_out.bpsp_residual)
            # self.set(filename, 'bpsp_base', loss_out.bpsp_base)
            # self.set(filename, 'bpsp_residual', loss_out.bpsp_residual)
        else:
            self.set(filename, 'loss', loss_out)

    def means_dict(self) -> Dict[str, float]:
        return {metric: float(np.mean(list(self.per_img_results[metric].values())))
                for metric in self.per_img_results.keys()}

    def per_image_results(self):
        res = defaultdict(list)
        metrics = sorted(list(self.per_img_results.keys()))
        for metric in metrics:
            for filename, result in self.per_img_results[metric].items():
                res[filename].append(result)

        return res, metrics

    def contains_metric(self, metric):
        return metric in self.per_img_results

    def sorted_values(self, *metrics):
        assert len(metrics) > 0
        filenames = sorted(self.per_img_results[metrics[0]].keys())
        return [(fn, *[self.per_img_results[metric][fn] for metric in metrics]) for fn in filenames]

    def means_str(self):
        return ' | '.join(f'{metric}: {mean:.3e}' for metric, mean in self.means_dict().items())

    def __str__(self):
        metrics_lens = ', '.join(f'{metric}: {len(self.per_img_results[metric].values())}'
                                 for metric in sorted(self.per_img_results.keys()))
        return f'TestResults({metrics_lens};means = {self.means_str()})'


class TestOutputCache(object):
    def __init__(self, test_log_dir):
        assert os.path.isdir(test_log_dir)
        self.test_log_dir = test_log_dir
        self.lock_file = os.path.join(test_log_dir, '.cache.lock')
        self.pickle_file = os.path.join(test_log_dir, CACHE_PKL)

    @contextmanager
    def _acquire_lock(self):
        with fasteners.InterProcessLock(self.lock_file):
            yield

    def __contains__(self, test_id):
        with self._acquire_lock():
            return test_id in self._read()

    def __setitem__(self, test_id, results: TestResults):
        with self._acquire_lock():
            cache = self._read()
            cache[test_id] = results
            self._write(cache)

    def __getitem__(self, test_id) -> TestResults:
        with self._acquire_lock():
            cache = self._read()
            return cache[test_id]

    def latest_cached(self):
        cache = self._read()
        if not cache:
            return None
        return max(test_id.restore_itr for test_id in cache.keys())

    def best_cached(self, metric, dataset_id) -> TestID:
        cache = self._read()
        best = None
        for test_id, value in cache.items():
            if test_id.dataset_id != dataset_id:
                continue
            means = value.means_dict()[metric]
            if not best or means < best[1]:
                best = test_id, means
        if not best:
            raise ValueError(f'Nothing found matching {metric} and {dataset_id} in {str(self)}')
        latest_itr = self.latest_cached()
        best_itr = best[0].restore_itr
        if best_itr != latest_itr:
            print(f'best_itr != latest_itr: {best_itr} != {latest_itr}')
        return best[0]

    def _read(self):
        if not os.path.isfile(self.pickle_file):
            return {}
        with open(self.pickle_file, 'rb') as f:
            return pickle.load(f)

    def _write(self, cache):
        with open(self.pickle_file, 'wb') as f:
            return pickle.dump(cache, f)

    def __str__(self):
        with self._acquire_lock():
            cache = self._read()
            return '\n'.join(f'{test_id}: {value}' for test_id, value in cache.items())


class CropMeans(object):
    """
    Helper class for the crop4 datasets: Since the crops may have different dimensions,
    we have to be careful when evaluating and do bpsp(img) = sum bits of crops / sum num subpixels of crops
    """
    def __init__(self):
        self.bits = collections.defaultdict(int)
        self.num_sub_pixels = collections.defaultdict(int)
        self.num_crops = collections.defaultdict(int)

        self.prefix_re = re.compile(r'(.*)_\d')

    def add(self, filename, n_sp_pre_pad, loss_out):
        if not isinstance(loss_out, EnhancementLoss):
            raise NotImplementedError(type(loss_out))
        prefix_match = self.prefix_re.match(filename)
        if not prefix_match:
            raise ValueError(f'Cannot match {filename}')
        prefix = prefix_match.group(1)
        self.num_crops[prefix] += 1
        self.bits[prefix] += (n_sp_pre_pad * loss_out.bpsp_base +
                              n_sp_pre_pad * loss_out.bpsp_residual).item()
        self.num_sub_pixels[prefix] += n_sp_pre_pad

    def means_dict(self):
        num_crops = set(self.num_crops.values())
        assert len(num_crops) == 1, (num_crops, self.num_crops)
        return {prefix: self.bits[prefix] / self.num_sub_pixels[prefix]
                for prefix in self.bits.keys()}

    def write_to_results(self, test_results: TestResults):
        for prefix, bpsp in self.means_dict().items():
            test_results.set(prefix, 'bpsp_crops', bpsp)


class QHistory(object):
    """
    Store for every image the optimal Q in a file named out_Dir/testid_restoreitr_logdate.csv
    """
    def __init__(self, log_date, test_id: TestID, out_dir='q_histories'):
        assert out_dir, out_dir
        self.test_id = test_id
        self.log_date = log_date
        self.cache = {}
        self.out_dir = out_dir

    def __setitem__(self, filename, q):
        self.cache[filename] = q

    @staticmethod
    def read_q_history(optimal_qs_csv):
        cache = {}
        with open(optimal_qs_csv, 'r') as csvfile:
            r = csv.reader(csvfile)
            next(r, None)  # skip first header
            for filename, q in r:
                if filename == 'image':
                    print('Ignoring another header in', optimal_qs_csv)
                    continue
                cache[filename] = torch.tensor(int(q))
        return cache

    def out_path(self):
        return os.path.join(
          self.out_dir, f'{self.test_id.dataset_id}_{self.test_id.restore_itr}_{self.log_date}.csv')

    def _get_p(self):
        os.makedirs(self.out_dir, exist_ok=True)
        return self.out_path()


    def read(self):
        p = self._get_p()
        if not os.path.isfile(p):
            return None
        with open(p, 'r') as csvfile:
            r = csv.reader(csvfile)
            for filename, q in r:
                self.cache[filename] = q
                yield filename

    def write_out(self):
        p = self._get_p()
        with open(p, 'w', newline='') as csvfile:
            w = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            w.writerow(['image', 'q'])
            for filename, q in sorted(self.cache.items()):
                w.writerow([filename, q])
        print('Created', p)


ITER_ALL_Q = int(os.environ.get('ITER_ALL_Q', 0)) == 1


class LossOutMeta(object):
    """
    Used to combine the loss for the meta dataset case.
    NOTE: we create an instance of this for every image. Contents are saved to memory, created using `create_memory`.
    """
    def __init__(self, is_meta, blueprint):
        """
        :param is_meta: either False or int. If int, represents the first Q to try (?)
        """
        self.is_meta = is_meta
        self.blueprint = blueprint
        self.n_sp_pre_pad = None
        self.losses = {}  # Q -> loss
        self.starting_q = is_meta
        if self.is_meta:
            assert isinstance(is_meta, int), is_meta
            self.next_q = is_meta
            self.clf = blueprint.clf
            self.qstrategy = blueprint.qstrategy
            self.run_clf = self.qstrategy in (QStrategy.CLF, QStrategy.CLF_ONLY)
        else:
            self.next_q = None
            self.clf = None
        self.clf_q = None

    def _unpack(self, img):
        """ Given a dictionary `img`, get x_n, bpps, n_sp_pre_pad """
        # TODO: only supported by EnhancementBlueprint
        if not isinstance(self.blueprint, EnhancementBlueprint):
            raise NotImplementedError()
        return self.blueprint.unpack_batch_light(img)  # returns raw, compressed, bpps
        # x_n, bpps, n_sp_pre_pad = self.blueprint.unpack_batch_pad(img, do_pad=False)
        # if self.n_sp_pre_pad:
        #     assert self.n_sp_pre_pad == n_sp_pre_pad
        # else:
        #     self.n_sp_pre_pad = n_sp_pre_pad
        # return x_n, bpps, n_sp_pre_pad

    # @staticmethod
    # def extract_raw(img):

    def unroll_unpack(self, img):
        """
        Main function. Given an image `img`. Unrolls _unpack over all needed Qs:

        - if the dataset is not meta:
            just return _unpack(image)
        - otherwise:
            if run_clf:
                set self.clf_q from classifier
            if not CLF_ONLY:
                do the parabola search.

        Always yields a set of images, and the q used to optain it.
        """
        if not self.is_meta:
            yield self._unpack(img), None
        else:
            # get a first Q guess
            if self.run_clf:
                assert self.clf
                # (x_r, _), _, _ = self._unpack(img[next(iter(img))])

                raw, _, _ = self._unpack(img[next(iter(img))])
                s_r = SymbolTensor(raw.long(), L=256)
                x_r = s_r.to_norm()

                torch.cuda.empty_cache()
                # crop_qs = [self.clf.get_q(x_r_crop) for x_r_crop in auto_crop.iter_crops(x_r.get())]
                # print('***\n',crop_qs,'\n')
                self.clf_q = self.clf.get_q(x_r.get())
                torch.cuda.empty_cache()
            elif self.qstrategy == QStrategy.FIXED:
                # note: we use clf_q also for the fixed Q, confusing naming scheme, I know!!!
                self.clf_q = 14  # optimal on training set

            # do optimum find
            if self.qstrategy != QStrategy.CLF_ONLY:
                while self.next_q and self.next_q in img:
                    # this is a dict with raw, compressed, bpps now
                    yield self._unpack(img[self.next_q]), self.next_q

            # make sure the guess was also evaluated
            if self.clf_q and self.clf_q not in self.losses:
                if self.clf_q not in img:
                    raise ValueError('**** CLF returned invalid q', self.clf_q)
                else:
                    # get this one as well
                    yield self._unpack(img[self.clf_q]), self.clf_q

    def append(self, loss_out, q):
        """
        Append a loss_out created in multiscale_tester to the cache.
        Sets self.next_q inplace!
        """
        self.losses[q] = loss_out
        if not q or ITER_ALL_Q:
            return
        try:
            if q == self.starting_q:  # first, go to the right.
                self.next_q = self.starting_q + 1
            elif q == self.starting_q + 1:  # went to the right from start, now:
                if enhancement_loss_lt(self.losses[self.starting_q + 1], self.losses[self.starting_q]):
                    # good, let's move into that direction
                    self.next_q = self.starting_q + 2
                else:
                    # nope, let's try the left side!
                    self.next_q = self.starting_q - 1
            elif q == self.starting_q + 2:
                if enhancement_loss_lt(self.losses[self.starting_q + 2], self.losses[self.starting_q + 1]):
                    self.next_q = self.starting_q + 3
                else:  # 14 is the best!
                    self.next_q = None
            elif q == self.starting_q - 1:
                if enhancement_loss_lt(self.losses[self.starting_q - 1], self.losses[self.starting_q]):
                    self.next_q = self.starting_q - 2
                else:
                    self.next_q = None
            elif q <= self.starting_q - 2:
                self.next_q = q - 1
            else:
                self.next_q = q + 1
        except KeyError as e:  # could not compare. Meaning we probably did not try all Q?
            if self.qstrategy != QStrategy.CLF_ONLY:
                print('*** Not found:', e)
            self.next_q = None

    @staticmethod
    def create_memory(is_meta, blueprint):
        """
        Create memory for metadatasets and qstrategy != CLF_ONLY
        Used to compare different Q
        """
        if is_meta and blueprint.clf is not None and blueprint.qstrategy != QStrategy.CLF_ONLY:
            return defaultdict(list)
        return None

    def memory_to_str(self, memory, joiner):
        if memory:
            otp = []
            for k, vs in memory.items():
                otp.append(f'{k}, {sum(vs)/len(vs):.3f}')
            return joiner.join(otp)
        return ''

    def get_min(self, memory=None):
        """
        :return the loss that is the best.
        If qstartegy != QStrategy.CLF_ONLY, memory is expected, and it will be updated with some statistics.

        """
        assert self.losses
        if not isinstance(next(iter(self.losses.values())), EnhancementLoss):
            return self.losses.pop(), None
        else:
            if ITER_ALL_Q:
                _print_order([loss_out.bpsp_base + loss_out.bpsp_residual for q, loss_out
                              in sorted(self.losses.items())], assert_max_flips=1)
            # Will be set to a tuple of loss_out, q
            loss_out_min = None
            for q, loss_out in self.losses.items():
                if not loss_out_min or enhancement_loss_lt(loss_out, loss_out_min[0]):
                    loss_out_min = loss_out, q
            assert loss_out_min is not None

            if self.clf_q and self.qstrategy != QStrategy.CLF_ONLY:
                if not memory:
                    raise ValueError('Memory expected!')
                if self.clf_q not in self.losses:
                    raise ValueError  # should not happen

                _, optimal_q = loss_out_min
                memory['eq'].append(optimal_q == self.clf_q)
                memory['close1'].append(abs(optimal_q - self.clf_q) <= 1)
                loss_out_clf_q = self.losses[self.clf_q]
                memory['bpsp'].append((loss_out_clf_q.bpsp_base + loss_out_clf_q.bpsp_residual).item())
            return loss_out_min


def _print_order(l, assert_max_flips):
    otp = ''
    flips = 0
    for aprev, a in zip(l, l[1:]):
        if aprev > a:
            sign = '>'
        elif aprev <= a:
            sign = '<'
        if otp and sign != otp[-1]:
            flips += 1
        otp += sign
    if flips > assert_max_flips:
        raise ValueError(l, otp)
    print(otp)


