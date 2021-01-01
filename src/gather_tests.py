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
import argparse
import csv
import os
import glob
import shutil
import time

import test_dataset_parser

# TODO breaks import chain
from test.test_helpers import TestOutputCache, get_test_log_dir_root, TestResults

DEFAULT_SUBFOLDERS = ('', 'AWS')


def write_results_csv(results, out_p, metrics, verbose=True):
    if verbose:
        print('Writing', out_p, '...')
    with open(out_p, 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        header = ['image', *metrics]
        print('Header:', header)
        w.writerow(header)
        for values in results:
            w.writerow(values)


class Gatherer(object):
    def __init__(self, flags, reset=False, verbose=True, dataset=None):

        self.flags = flags
        self.verbose = verbose
        self.metric = 'bpsp'

        if not dataset:
            dataset = list(test_dataset_parser.parse_into_datasets(flags.images))
            assert len(dataset) == 1, dataset  # multiple datasets in your tester? nope not happening
            self.dataset = dataset[0]
            _, compressed_ds_dir = test_dataset_parser.name_from_images(flags.images)
        else:
            self.dataset, compressed_ds_dir = dataset

        self.test_log_dir_root = get_test_log_dir_root(flags.log_dir)
        csv_root_dir = self.test_log_dir_root.rstrip(os.path.sep) + '_csv'
        self.csv_dataset_dir = os.path.join(csv_root_dir, os.path.basename(compressed_ds_dir))

        if reset:
            print('*** rm -rf', self.csv_dataset_dir)
            time.sleep(2)
            shutil.rmtree(self.csv_dataset_dir)

        if verbose:
            print('Saving to', self.csv_dataset_dir)

        os.makedirs(self.csv_dataset_dir, exist_ok=True)

    def get_experiment_dir(self, log_date, subfolders=DEFAULT_SUBFOLDERS):
        for subfolder in subfolders:
            experiment_dir_glob = os.path.join(self.test_log_dir_root, subfolder, log_date + '*')
            try:
                return glob.glob(experiment_dir_glob)[0]
            except IndexError as e:
                print('No match for', experiment_dir_glob, '...')
        return None

    def gather_best(self):
        log_dates = self.flags.log_dates.split(',')
        for log_date in log_dates:
            experiment_dir = self.get_experiment_dir(log_date)
            if not experiment_dir:
                continue

            c = TestOutputCache(experiment_dir)
            best_id = c.best_cached(self.metric, self.dataset.id)
            best_results = c[best_id]
            if self.verbose:
                print(best_results)
            self.write(best_results, log_date, best_id.restore_itr)

    def write(self, results: TestResults, log_date, restore_itr):
        out_p = os.path.join(self.csv_dataset_dir, f'{log_date}.{restore_itr}.csv')
        metrics = [self.metric]
        if results.contains_metric('Q'):
            metrics.append('Q')
        write_results_csv(results.sorted_values(*metrics), out_p, metrics=metrics, verbose=self.verbose)



def main():
    p = argparse.ArgumentParser()

    p.add_argument('log_dir', help='Directory of experiments. Will create a new folder, LOG_DIR_test, to save test '
                                   'outputs.')
    p.add_argument('log_dates', help='A comma-separated list, where each entry is a log_date, such as 0104_1345. '
                                     'These experiments will be tested.')
    p.add_argument('images', help='A comma-separated list, where each entry is either a directory with images or '
                                  'the path of a single image OR a ;-separated raw;compressed pair. '
                                  'Will test on all these images.')

    p.add_argument('--reset', action='store_true')

    flags = p.parse_args()
    Gatherer(flags, reset=flags.reset).gather_best()



if __name__ == '__main__':
    main()
