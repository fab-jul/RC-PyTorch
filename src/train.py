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
import subprocess

import torch

# seed at least the random number generators.
# doesn't guarantee full reproducability: https://pytorch.org/docs/stable/notes/randomness.html
from test.multiscale_tester import MultiscaleTester
from train.classifier_trainer import ClassifierTrainer

torch.manual_seed(0)

# ---

import os
import argparse
import sys

import torch.backends.cudnn
from fjcommon import no_op

import pytorch_ext as pe
import math
from helpers import cpu_gpu_info
from helpers.configs_repo_setup import ConfigsRepo
from helpers.config_checker import DEFAULT_CONFIG_DIR
from helpers.global_config import global_config
from helpers.saver import Saver
from train.multiscale_trainer import MultiscaleTrainer
from train.train_restorer import TrainRestorer
from train.trainer import LogConfig

torch.backends.cudnn.benchmark = True


LOGS = os.environ.get('QSUB_LOGS', os.path.expanduser('~/phd_code/jointcomp/qsub_logs'))


def parse_num_workers_flag(N):
    if N == 'fair':
        return max(math.floor(cpu_gpu_info.get_num_CPUs() / cpu_gpu_info.get_num_GPUs()), 1) * 2
    return int(N)


# TODO: cleanup
def _print_debug_info():
    print('*' * 80)
    print('Running on {}'.format(pe.DEVICE))
    print('*' * 80)


def _log_date_from_log_id(job_id):
    assert os.path.isdir(LOGS), LOGS

    cmd = "grep -m 1 -r JOB_INFO %s | grep %s | awk -F';' '{print $3}'" % (LOGS, job_id)
    log_date = subprocess.check_output(cmd, shell=True).decode().strip()
    return log_date or None


def _update_flags_from_experiment(flags, log_dir_root, log_date_or_job_id, add_continue_flag):
    if '_' not in log_date_or_job_id:
        print('Parsing', log_date_or_job_id, 'as a JOB_ID')
        log_date = _log_date_from_log_id(job_id=log_date_or_job_id)
        if not log_date:
            raise ValueError(f'Cannot find LOG_DATE for JOB_ID={log_date_or_job_id}')
    else:
        log_date = log_date_or_job_id

    config_ps, experiment_dir = MultiscaleTester.get_configs_experiment_dir(log_dir_root, log_date,
                                                                            update_global_config=True)

    flags.ms_config_p, flags.dl_config_p = config_ps
    print(f'Did set '
          f'ms_config_p={flags.ms_config_p}, '
          f'dl_config_p={flags.dl_config_p}.')

    print(f'Did set global_config={global_config}')

    flags.description = f'RESTORE {log_date}' + (
        f'; {flags.description}' if flags.description else '')
    print(f'--description="{flags.description}"')

    print(f'--restore {log_date}')
    flags.restore = log_date

    if not flags.restore_restart and add_continue_flag:
        print(f'--restore_continue')
        flags.restore_continue = True


def main(args, configs_dir=DEFAULT_CONFIG_DIR):
    p = argparse.ArgumentParser()

    # TODO: describe
    p.add_argument('ms_config_p', help='Path to a multiscale config, see README')
    p.add_argument('dl_config_p', help='Path to a dataloader config, see README')
    p.add_argument('log_dir_root', default='logs', help='All outputs (checkpoints, tensorboard) will be saved here.')
    p.add_argument('--temporary', '-t', action='store_true',
                   help='If given, outputs are actually saved in ${LOG_DIR_ROOT}_TMP.')
    p.add_argument('--log_train', '-ltrain', type=int, default=100,
                   help='Interval of train output.')
    p.add_argument('--log_train_heavy', '-ltrainh', type=int, default=5, metavar='LOG_HEAVY_FAC',
                   help='Every LOG_HEAVY_FAC-th time that i %% LOG_TRAIN is 0, also output heavy logs.')
    p.add_argument('--log_val', '-lval', type=int, default=500,
                   help='Interval of validation output.')
    p.add_argument('--log_validation_set', '-lvals', type=float, default=60,
                   help='Interval in minutes of when to do entire validation set.')

    p.add_argument('--description', '-d', type=str,
                   help='Description, if given, is appended to Google Sheets.')

    p.add_argument('-p', action='append', nargs=1,
                   help='Specify global_config parameters, see README')

    p.add_argument('--restore', type=str, metavar='RESTORE_DIR',
                   help='Path to the log_dir of the model to restore. If a log_date ('
                        'MMDD_HHmm) is given, the model is assumed to be in LOG_DIR_ROOT.')
    p.add_argument('--restore_continue', action='store_true',
                   help='If given, continue in RESTORE_DIR instead of starting in a new folder.')
    p.add_argument('--restore_restart', action='store_true',
                   help='If given, start from iteration 0, instead of the iteration of RESTORE_DIR. '
                        'Means that the model in RESTORE_DIR is used as pretrained model')
    p.add_argument('--restore_itr', '-i', type=int, default=-1,
                   help='Iteration to restore from.')
    p.add_argument('--restore_strict', type=str, help='y|n', choices=['y', 'n'], default='y')

    p.add_argument('--num_workers', '-W', type=str, default='8',
                   help='Number of workers used for DataLoader')

    p.add_argument('--saver_keep_tmp_itr', '-si', type=int, default=250,
                   help='keep checkpoint every `keep_tmp_itr` iterations.')
    p.add_argument('--saver_keep_every', '-sk', type=int, default=10,
                   help='Keep every `keep_every`-th checkpoint, making it a persistent checkpoint.')
    p.add_argument('--saver_keep_tmp_last', '-skt', type=int, default=3,
                   help='Also keep the last `keep_tmp_last` temporary checkpoints before a persistent checkpoint.')

    p.add_argument('--no_saver', action='store_true',
                   help='If given, no checkpoints are stored.')

    p.add_argument('--debug', action='store_true')

    flags = p.parse_args(args)

    configs_repo = ConfigsRepo(configs_dir)  # make sure configs used for the experiment to restore also exist

    if flags.ms_config_p == 'RESTORE' or flags.ms_config_p == 'RESTORE_new':
        _update_flags_from_experiment(flags, flags.log_dir_root, flags.dl_config_p,
                                      add_continue_flag=flags.ms_config_p == 'RESTORE')

    _print_debug_info()

    if flags.debug:
        flags.temporary = True

    # Add to global config
    global_config.add_from_flag(flags.p)
    print(global_config)

    # make sure configs are available
    configs_repo.check_configs_available(flags.ms_config_p, flags.dl_config_p)

    num_workers = parse_num_workers_flag(flags.num_workers)
    print('Taskset: {} // num_workers: {}'.format(cpu_gpu_info.get_taskset(), num_workers))

    saver = (Saver(flags.saver_keep_tmp_itr, flags.saver_keep_every, flags.saver_keep_tmp_last,
                   verbose=True)
             if not flags.no_saver
             else no_op.NoOp())

    restorer = TrainRestorer.from_flags(flags.restore, flags.log_dir_root, flags.restore_continue, flags.restore_itr,
                                        flags.restore_restart, flags.restore_strict)

    log_config = LogConfig(flags.log_train, flags.log_val, flags.log_train_heavy, flags.log_validation_set)
    trainer_style = MultiscaleTrainer.get_style_from_config(flags.ms_config_p)
    trainer_cls = ClassifierTrainer if trainer_style == 'classifier' else MultiscaleTrainer
    trainer = trainer_cls(flags.ms_config_p, flags.dl_config_p,
                          flags.log_dir_root.rstrip(os.path.sep) + ('_TMP' if flags.temporary else ''),
                          log_config,
                          num_workers,
                          saver=saver, restorer=restorer,
                          )
    if not flags.debug:
        trainer.train()
    else:
        trainer.debug()




if __name__ == '__main__':
    main(sys.argv[1:])
