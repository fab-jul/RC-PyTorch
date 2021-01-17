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

--------------------------------------------------------------------------------

General Trainer class, subclassed by MultiscaleTrainer

Every VAL_INTERVAL minutes, we go into validation, where we evaluate the *entire* validation set.
The average bpsp is then logged to tensorboard and to TestResults (?)

"""
import time
from collections import namedtuple

import torch
import os

import torchvision
from fjcommon import timer
from fjcommon.no_op import NoOp

from helpers import logdir_helpers
import vis.safe_summary_writer
from helpers.persistent_random_sampler import PersistentRandomSampler
from helpers.saver import Saver
from helpers.global_config import global_config
import itertools

from train.train_restorer import TrainRestorer


LogConfig = namedtuple('LogConfig', ['log_train', 'log_val', 'log_train_heavy', 'log_validation_set'])


class TimedIterator(object):
    def __init__(self, it):
        self.timer = timer.TimeAccumulator()
        self.it = iter(it)

    def mean_time_spent(self):
        return self.timer.mean_time_spent()

    def __iter__(self):
        return self

    def __next__(self):
        with self.timer.execute():
            return next(self.it)

# TODO: allow "restart last epoch" or sth
class TrainingSetIterator(object):
    """ Implements skipping to a certain iteration """
    def __init__(self, skip_to_itr, dl_train):
        """
        :param skip_to_itr: Iteration (index of the batch) to skip to
        :param dl_train: DataLoader to use
        """
        self.skip_to_itr = skip_to_itr
        self.dl_train = dl_train
        self.epoch_len = len(self.dl_train)

        self.sampler = self.dl_train.sampler
        assert isinstance(self.sampler, PersistentRandomSampler), type(self.sampler)

    def iter_epoch(self, epoch):
        """ :returns an iterator over tuples (itr, batch) """
        num_epochs_to_skip, num_batches_to_skip = self._get_skips()
        if epoch < num_epochs_to_skip:
            print(f'*** Skipping epoch {epoch}')
            return []  # nothing to iterate

        # Make sure smapler knows current epoch
        self.sampler.current_epoch = epoch
        # Resets the line below, where num_items_to_skip is set. Cannot reset it down there because the value is NOT
        # used when iter(self.dl_train) is called. instead, it is read when __iter__ of PersistentRandomSampler is
        # called, which seems to be the point where the data-iter loop is actually executed.
        self.sampler.num_items_to_skip = 0
        if epoch > num_epochs_to_skip or (epoch == num_epochs_to_skip and num_batches_to_skip == 0):
            # iterate like normal, i.e., we have nothing to skip in this epoch
            return enumerate(self.dl_train, epoch * self.epoch_len)

        print(f'*** Epoch {epoch}: Skipping {num_batches_to_skip} batches...')
        # if we get to here, we are in the first epoch of which we should skip part of, so skip
        # `num_batches_to_skip` batches.
        self.sampler.num_items_to_skip = num_batches_to_skip * self.dl_train.batch_size
        it = iter(self.dl_train)
        return enumerate(it, epoch * self.epoch_len + num_batches_to_skip)

    def _get_skips(self):
        """ Split skip_to_itr into entire epochs to skip and batches to skip in the last epoch. """
        if self.skip_to_itr:
            num_epochs_to_skip = self.skip_to_itr // self.epoch_len
            num_batches_to_skip = self.skip_to_itr % self.epoch_len
            return num_epochs_to_skip, num_batches_to_skip
        return 0, 0


class AbortTrainingException(Exception):
    pass


class Trainer(object):
    def __init__(self, dl_train, optims, net, sw: vis.safe_summary_writer.SafeSummaryWriter,
                 max_epochs, log_config: LogConfig, saver: Saver=None, skip_to_itr=None):

        assert isinstance(optims, list)

        self.dl_train = dl_train
        self.dl_val_itr = None
        self.optims = optims
        self.net = net
        self.sw = sw
        self.max_epochs = max_epochs
        self.log_config = log_config
        self.saver = saver if saver is not None else NoOp

        self.skip_to_itr = skip_to_itr

    @staticmethod
    def print_job_info(log_date, description):
        if not description:
            description = ''
        print(f'JOB_INFO ; {os.environ.get("JOB_ID", "")} ; {log_date} ; {description}')

    def continue_from(self, ckpt_dir):
        pass

    def train(self):
        log_train, log_val, log_train_heavy, log_validation_set = self.log_config

        dl_train_it = TrainingSetIterator(self.skip_to_itr, self.dl_train)

        _print_unused_global_config()

        try:
            last_validation_set_test = time.time()
            validation_set_interval_s = log_validation_set * 60
            if int(os.environ.get('VAL_AT_BEGINNING', 0)):
                print('*** Validating right away...')
                last_validation_set_test -= 10 * validation_set_interval_s
            for epoch in (range(self.max_epochs) if self.max_epochs else itertools.count()):
                self.print_epoch_sep(epoch)
                self.prepare_for_epoch(epoch)
                t = TimedIterator(dl_train_it.iter_epoch(epoch))
                for i, img_batch in t:
                    for o in self.optims:
                        o.zero_grad()
                    should_log = (i > 0 and i % log_train == 0)
                    should_log_heavy = (i > 0 and (i / log_train_heavy) % log_train == 0)
                    self.train_step(i, img_batch,
                                    log=should_log,
                                    log_heavy=should_log_heavy,
                                    load_time_per_batch=t.mean_time_spent() if should_log else None)
                    self.saver.save(self.modules_to_save(), i)
                    if i > 0 and log_val > 0 and i % log_val == 0:
                        self._eval(i, kind='fixed_first')
                    if i % 1000 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    if time.time() - last_validation_set_test > validation_set_interval_s:
                        print('Validating...')
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.saver.save(self.modules_to_save(), i, force=True, make_permanent=True)
                        self._eval(i, kind='validation_set')
                        last_validation_set_test = time.time()
        except AbortTrainingException as e:
            print('Caught {}'.format(e))
            return

    def _eval(self, i, kind):
        self.net.eval()
        with torch.no_grad():
            self.validation_step(i, kind)
        self.net.train()

    def validation_step(self, i, kind):
        raise NotImplementedError

    def debug(self):
        print('Debug ---')
        _print_unused_global_config()
        self.prepare_for_epoch(0)
        self.train_step(0, next(iter(self.dl_train)),
                        log=True, log_heavy=True, load_time_per_batch=0)
        self._eval(101)

    def print_epoch_sep(self, epoch):
        print('-' * 80)
        print(' EPOCH {}'.format(epoch))
        print('-' * 80)

    def modules_to_save(self):
        """ used to save and restore. Should return a dictionary module_name -> nn.Module """
        raise NotImplementedError()

    def train_step(self, i, img_batch, log, log_heavy, load_time_per_batch=None):
        raise NotImplementedError()

    def validation_loop(self, i):
        raise NotImplementedError()

    def prepare_for_epoch(self, epoch):
        pass

    def add_filter_summaray(self, tag, p, global_step):
        if len(p.shape) == 1:  # bias
            C, = p.shape
            p = p.reshape(C, 1).expand(C, C).reshape(C, C, 1, 1)

        try:
            _, _, H, W = p.shape
        except ValueError:
            if global_step == 0:
                print('INFO: Cannot unpack {} ({})'.format(p.shape, tag))
            return

        if H == W == 1:  # 1x1 conv
            p = p[:, :, 0, 0]
            filter_vis = torchvision.utils.make_grid(p, normalize=True)
            self.sw.add_image(tag, filter_vis, global_step)

    @staticmethod
    def update_lrs(epoch, optims_lrs_facs_interval):
        raise DeprecationWarning('use lr_schedule.py')
        for optimizer, init_lr, decay_fac, interval_epochs in optims_lrs_facs_interval:
            if decay_fac is None:
                continue
            Trainer.exp_lr_scheduler(optimizer, epoch, init_lr, decay_fac, interval_epochs)

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr, decay_fac=0.1, interval_epochs=7):
        raise DeprecationWarning('use lr_schedule.py')
        lr = init_lr * (decay_fac ** (epoch // interval_epochs))
        print('LR = {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_lrs(self):
        for optim in self.optims:
            for param_group in optim.param_groups:
                yield param_group['lr']

    def maybe_restore(self, restorer: TrainRestorer):
        """
        :return: skip_to_itr
        """
        if restorer is None:
            return None  # start from 0
        restore_itr = restorer.restore_desired_ckpt(self.modules_to_save())  # TODO: allow arbitrary ckpts
        if restorer.restart_at_zero:
            return 0
        return restore_itr

    @staticmethod
    def get_log_dir(log_dir_root, rel_paths, restorer, strip_ext='.cf', global_config_values=None):
        if not restorer or not restorer.restore_continue:
            log_dir = logdir_helpers.create_unique_log_dir(
                    rel_paths, log_dir_root, strip_ext=strip_ext, postfix=global_config_values)
            print('Created {}...'.format(log_dir))
            return log_dir

        previous_log_dir = restorer.get_log_dir()

        if restorer.restore_continue:
            theoretical_log_dir = logdir_helpers.get_log_dir_name(
                    rel_paths, strip_ext=strip_ext, postfix=global_config_values)
            previous_log_dir_name = logdir_helpers.log_name_from_log_dir(previous_log_dir)
            if theoretical_log_dir != previous_log_dir_name and not global_config.get('force_overwrite', False):
                raise ValueError('--restore_continue given, but previous log_dir != current:\n' +
                                 f'   {previous_log_dir_name}\n!= {theoretical_log_dir}')

        print('Using {}...'.format(previous_log_dir))
        return previous_log_dir


def _print_unused_global_config(ignore=None):
    """ For safety, print parameters that were passed with -p but never used during construction of graph. """
    if not ignore:
        ignore = []
    unused = [u for u in global_config.get_unused_params() if u not in ignore]
    if unused:
        raise ValueError('Unused params:\n- ' + '\n- '.join(unused))

