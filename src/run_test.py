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

--------------------------------------------------------------------------------

This code uses a cache: If some experiment has already been tested for some iteration and crop and dataset,
we just print that (see TestID in multiscale_tester.py).

"""
from collections import defaultdict
from typing import List, Dict, Tuple
import time
import os

import torch.backends.cudnn

from blueprints import enhancement_blueprint
from dataloaders.compressed_images_loader import is_residual_dataset
from helpers.global_config import global_config
from helpers.logdir_helpers import is_log_date
from helpers.saver import CkeckpointLoadingException
from test_dataset_parser import parse_into_datasets

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

import sys
import argparse

from helpers.aligned_printer import AlignedPrinter
from test.multiscale_tester import MultiscaleTester, MultiscaleTesterInitException, GLOBAL_CONFIG_TEST_PREFIX


def dict_itemgetter(key, default_key):
    def _getter(d: dict):
        return d.get(key, d[default_key])
    return _getter


def _to_str(e):
    if isinstance(e, int):
        return str(e)
    if isinstance(e, float):
        if 0 < e < 10:
            return f'{e:.3f}'
        return f'{e:.3e}'
    # print(type(e))
    return e


# For notebook
def get_tester_and_dataset(args):
    flags, datasets, dataset_style, log_dates, log_date_to_names = parse_flags(args)
    return next(iter_testers(flags, log_dates, configs_dir=None, dataset_style=dataset_style)), datasets[0]


def main(args=sys.argv[1:], configs_dir=None):
    # Fixes crashes on jobs from AWS...
    if 'DATA_ROOT' not in os.environ:
        os.environ['DATA_ROOT'] = ''

    flags, datasets, dataset_style, log_dates, log_date_to_names = parse_flags(args)

    results: List[Dict[str, object]] = []
    for tester in iter_testers(flags, log_dates, configs_dir, dataset_style=dataset_style):
        # print(tester.test_output_cache)
        # continue
        if not flags.write_to_files:
            results += tester.test_all(datasets)
        else:
            results += tester.write_to_files(datasets)

    print_order = ['testset', 'exp', 'itr']

    if results:
        print('*** Summary:')

        points_ours: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)  # dict: testset -> List[str, float, flat]

        with AlignedPrinter() as a:
            # Results is a list of dicts. We sort it by the key given by sort_output.
            # for every result, that does *not* have this key, sort that by print_order[0]
            for result in sorted(results, key=dict_itemgetter(flags.sort_output, print_order[0])):
                for k in sorted(result.keys()):
                    if k not in print_order:
                        print_order.append(k)
                # now, print_order contains all keys
                result['exp'] = log_date_to_names[result['exp']]
                # add to printer, make sure it's all strings
                a.append(list(map(_to_str, (result.get(k, '')
                                            for k in print_order))))
                # if flags.plot_psnr and 'bpp_zs' in result and 'psnr_rgb' in result:
                #     points_ours[result['testset']].append((result['exp'], result['bpp_zs'], result['psnr_rgb']))
            # add header
            a.insert(0, print_order)

#        if points_ours:
#            for testset_id, values in points_ours.items():
#                out_dir = os.path.join(flags.plot_psnr, testset_id)
#                if not os.path.isdir(out_dir):
#                    print(f'*** Expected {out_dir} to exist, for plotting')
#                    continue
#                other_codecs.plot_measured_dataset(out_dir, {'psnr': values})


def _parse_names_file(names_p, target: set):
    """
    :param names_p: path to names file
    :param target:  names to find, as set of log_dates or set of names
    :return: log_dates and corresponding names
    """
    with open(names_p, 'r') as f:
        log_dates, names = [], []
        for l in f:
            if not l.strip():  # empty lines
                continue
            l = l.split('#')[0]
            log_date, name = (e.strip() for e in l.split(':'))
            if name in target or log_date in target:
                try:
                    target.remove(log_date)
                except KeyError:
                    target.remove(name)
                log_dates.append(log_date)
                names.append(name)
    for t in target:  # remaining entries in target
        if not is_log_date(t):
            raise ValueError(f'Did not find {t} in {names_p}')
        log_dates.append(t)
        names.append(t)  # just call it by the log date
    return ','.join(log_dates), ','.join(names)


def parse_flags(args):
    p = argparse.ArgumentParser()

    p.add_argument('log_dir', help='Directory of experiments. Will create a new folder, LOG_DIR_test, to save test '
                                   'outputs.')
    p.add_argument('log_dates', help='A comma-separated list, where each entry is a log_date, such as 0104_1345. '
                                     'These experiments will be tested.')
    p.add_argument('images', help='A comma-separated list, where each entry is either a directory with images or '
                                  'the path of a single image OR a ;-separated raw;compressed pair. '
                                  'Will test on all these images.')
    p.add_argument('--match_filenames', '-fns', nargs='+', metavar='FILTER',
                   help='If given, remove any images in the folders given by IMAGES that do not match any '
                        'of specified filter.')
    p.add_argument('--max_imgs_per_folder', '-m', type=int, metavar='MAX',
                   help='If given, only use MAX images per folder given in IMAGES. Default: None')
    p.add_argument('--crop', type=int, help='Crop all images to CROP x CROP squares. Default: None')

    p.add_argument('--names', '-n', type=str,
                   help='Comma separated list, if given, must be as long as LOG_DATES. Used for output. If not given, '
                        'will just print LOG_DATES as names. If a file ending in .txt is given, it must be a : table.')

    p.add_argument('--clf_p', type=str, help="Path to the ckpt to use for the Q classifier.")
    p.add_argument('--qstrategy', choices=[q.value for q in enhancement_blueprint.QStrategy])
    p.add_argument('--tau_optimization', action='store_true')


    p.add_argument('--modulo_op', type=int)

    p.add_argument('--overwrite_cache', '-f', action='store_true',
                   help='Ignore cached test outputs, and re-create.')
    p.add_argument('--reset_entire_cache', action='store_true',
                   help='Remove cache.')

    p.add_argument('--restore_itr', '-i', default='-1',
                   help='Which iteration to restore. -1 means latest iteration. Will use closest smaller if exact '
                        'iteration is not found. Can be "latest_cached", but that depends on testsets. Can be 100k.'
                        'Default: -1')

    # p.add_argument('--recursive', default='0',
    #                help='Either an number or "auto". If given, the rgb configs with num_scales == 1 will '
    #                     'automatically be evaluated recursively (i.e., the RGB baseline). See _parse_recursive_flag '
    #                     'in multiscale_tester.py. Default: 0')

    p.add_argument('--sample', type=str, metavar='SAMPLE_OUT_DIR',
                   help='Sample from model. Store results in SAMPLE_OUT_DIR.')

    # p.add_argument('--plot_psnr', type=str, metavar='ROOT_OTHER_CODECS_DIR')

    p.add_argument('--write_means', type=str, metavar='WRITE_OUT_DIR')  # TODO; is implemented actually

    p.add_argument('--write_to_files', type=str, metavar='WRITE_OUT_DIR',
                   help='Write images to files in folder WRITE_OUT_DIR, with arithmetic coder. If given, the cache is '
                        'ignored and no test output is printed. Requires torchac to be installed, see README. Files '
                        'that already exist in WRITE_OUT_DIR are overwritten.')
    p.add_argument('--compare_theory', action='store_true',
                   help='If given with --write_to_files, will compare actual bitrate on disk to theoretical bitrate '
                        'given by cross entropy.')

    p.add_argument('--save_imgs', action='store_true')

    p.add_argument('--gather', action='store_true')

    p.add_argument('-p', action='append',
                   help='Specify test_config parameters')

    p.add_argument('--time_report', type=str, metavar='TIME_REPORT_PATH',
                   help='If given with --write_to_files, write a report of time needed for each component to '
                        'TIME_REPORT_PATH.')

    p.add_argument('--sort_output', '-s', default='testset',
                   help='How to sort the final summary. Default: testset')

    flags = p.parse_args(args)

    if flags.p:
        assert all(p.startswith(GLOBAL_CONFIG_TEST_PREFIX) for p in flags.p), GLOBAL_CONFIG_TEST_PREFIX
        global_config.add_from_flag(flags.p)
        print('Global config\n', global_config)


    if flags.clf_p:
        assert os.path.isfile(flags.clf_p), flags.clf_p

    if flags.compare_theory and not flags.write_to_files:
        raise ValueError('Cannot have --compare_theory without --write_to_files.')
    if flags.write_to_files and flags.sample:
        raise ValueError('Cannot have --write_to_files and --sample.')
    if flags.time_report and not flags.write_to_files:
        raise ValueError('--time_report only valid with --write_to_files.')
    # if flags.plot_psnr and not os.path.isdir(flags.plot_psnr):
    #     raise ValueError('Directory specified by --plot_psnr does not exist.')

    # log_dates can also be a .txt file with names
    # if not flags.log_dates.endswith('.txt'):
    #     flags.names = flags.log_dates
    if flags.names and flags.names.endswith('.txt'):
        flags.log_dates, flags.names = _parse_names_file(flags.names, set(flags.log_dates.split(',')))

    splitter = ',' if ',' in flags.log_dates else '|'  # support tensorboard strings, too
    log_dates = flags.log_dates.split(splitter)
    assert len(log_dates) == len(set(log_dates)), f'Contains duplicates: {log_dates}\nUse {",".join(set(log_dates))}'

    # def _lift(images_dir_or_image):
    #     if images_dir_or_image != 'checkerboard':
    #         return images_dir_or_image
    #     return CheckerboardTestset()

    datasets = list(parse_flags_into_datasets(flags))
    dataset_style = get_dataset_style(datasets)
    print(f'Got {len(datasets)} datasets.')

    # if --names was passed: will print 'name (log_date)'. otherwise, will just print 'log_date'
    if flags.names:
        names = flags.names.split(splitter) if flags.names else log_dates
        log_date_to_names = {log_date: f'{name} ({log_date})'
                             for log_date, name in zip(log_dates, names)}
    else:
        # set names to log_dates if --names is not given, i.e., we just output log_date
        log_date_to_names = {log_date: log_date for log_date in log_dates}

    return flags, datasets, dataset_style, log_dates, log_date_to_names


def get_dataset_style(datasets):
    if any(map(is_residual_dataset, datasets)):
        return 'enhancement'
    # if any(map(is_classifier_dataset, datasets)):
    #     return 'classifier'
    return 'multiscale'


def parse_flags_into_datasets(flags):
    return parse_into_datasets(flags.images, flags.crop, flags.match_filenames, flags.max_imgs_per_folder)


def iter_testers(flags, log_dates, configs_dir, dataset_style):
    # doing this so that doing run_test 200k,300k,400k does not re-eval 3 times
    before_time = time.time()

    for log_date in log_dates:
        for restore_itr in flags.restore_itr.split(','):
            print('Testing {} at {} ---'.format(log_date, restore_itr))
            try:
                yield MultiscaleTester(log_date, flags, restore_itr,
                                       configs_dir=configs_dir,
                                       style=dataset_style,
                                       filter_ckpts_at=before_time)
            except MultiscaleTesterInitException as e:
                print('*** Error while initializing:', e)
            except CkeckpointLoadingException as e:
                print('*** Error while restoring:', e)


if __name__ == '__main__':
    main()
