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

Main trainer class. Called multiscae_Trainer for legacy reasons.

"""
import os
import numpy as np
from collections import defaultdict

from fjcommon import config_parser
from fjcommon import functools_ext as ft
from fjcommon import timer
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

import vis.summarizable_module
from blueprints import shared
from blueprints.classifier_blueprint import ClassifierBlueprint
from dataloaders.cached_listdir_imgs import cached_listdir_imgs
from dataloaders.checkerboard import make_checkerboard_dataset, _CheckerboardDataset
from dataloaders.compressed_images_loader import get_residual_dataset
from dataloaders.images_loader import IndexDataset
from helpers import logdir_helpers
import vis.safe_summary_writer
from blueprints.enhancement_blueprint import EnhancementBlueprint
from dataloaders import images_loader

from helpers.persistent_random_sampler import PersistentRandomSampler
from helpers.global_config import global_config
from helpers.paths import CKPTS_DIR_NAME
from helpers.saver import Saver
from test import multiscale_tester
from test.test_helpers import TestID, TestResults
from train import lr_schedule
from train.train_restorer import TrainRestorer
from train.trainer import LogConfig, Trainer


class MultiscaleTrainer(Trainer):
    def __init__(self,
                 config_p, dl_config_p,
                 log_dir_root, log_config: LogConfig,
                 num_workers,
                 saver: Saver, restorer: TrainRestorer=None,
                 sw_cls=vis.safe_summary_writer.SafeSummaryWriter):
        """
        :param config_p: Path to the network config file, see README
        :param dl_config_p: Path to the dataloader config file, see README
        :param log_dir_root: All outputs (checkpoints, tensorboard) will be saved here.
        :param log_config: Instance of train.trainer.LogConfig, contains intervals.
        :param num_workers: Number of workers to use for DataLoading, see train.py
        :param saver: Saver instance to use.
        :param restorer: Instance of TrainRestorer, if we need to restore
        """
        self.style = MultiscaleTrainer.get_style_from_config(config_p)
        self.blueprint_cls = {'enhancement': EnhancementBlueprint,
                              'classifier': ClassifierBlueprint}[self.style]

        global_config.declare_used('filter_imgs')

        # Read configs
        # config = config for the network
        # config_dl = config for data loading
        (self.config, self.config_dl), rel_paths = ft.unzip(map(config_parser.parse, [config_p, dl_config_p]))
        # TODO only read by enhancement classes
        self.config.is_residual = self.config_dl.is_residual_dataset

        # Update global_config given config.global_config
        global_config_config_keys = global_config.add_from_str_without_overwriting(self.config.global_config)
        # Update config_ms depending on global_config
        global_config.update_config(self.config)

        if self.style == 'enhancement':
            EnhancementBlueprint.read_evenly_spaced_bins(self.config_dl)

        self._custom_init()

        # Create data loaders
        dl_train, self.ds_val, self.fixed_first_val = self._get_dataloaders(num_workers)
        # Create blueprint. A blueprint collects the network as well as the losses in one class, for easy reuse
        # during testing.
        self.blueprint = self.blueprint_cls(self.config)
        print('Network:', self.blueprint.net)
        # Setup optimizer
        optim_cls = {'RMSprop': optim.RMSprop,
                     'Adam': optim.Adam,
                     'SGD': optim.SGD,
                     }[self.config.optim]
        net = self.blueprint.net
        self.optim = optim_cls(net.parameters(), self.config.lr.initial,
                               weight_decay=self.config.weight_decay)
        # Calculate a rough estimate for time per batch (does not take into account that CUDA is async,
        # but good enought to get a feeling during training).
        self.time_accumulator = timer.TimeAccumulator()
        # Restore network if requested
        skip_to_itr = self.maybe_restore(restorer)
        if skip_to_itr is not None:  # i.e., we have a restorer
            print('Skipping to {}...'.format(skip_to_itr))
        # Create LR schedule to update parameters
        self.lr_schedule = lr_schedule.from_spec(
                self.config.lr.schedule, self.config.lr.initial, [self.optim], epoch_len=len(dl_train))

        # --- All nn.Modules are setup ---
        print('-' * 80)

        # create log dir and summary writer
        self.log_dir_root = log_dir_root
        global_config_values = global_config.values(ignore=global_config_config_keys)
        self.log_dir = Trainer.get_log_dir(log_dir_root, rel_paths, restorer,
                                           global_config_values=global_config_values)
        self.log_date = logdir_helpers.log_date_from_log_dir(self.log_dir)
        self.ckpt_dir = os.path.join(self.log_dir, CKPTS_DIR_NAME)
        print(f'Checkpoints will be saved to {self.ckpt_dir}')
        saver.set_out_dir(self.ckpt_dir)

        if global_config.get('ds_syn', None):
            underlying = dl_train.dataset
            while not isinstance(underlying, _CheckerboardDataset):
                underlying = underlying.ds
            underlying.save_all(self.log_dir)


        # Create summary writer
        sw = sw_cls(self.log_dir)
        self.summarizer = vis.summarizable_module.Summarizer(sw)
        net.register_summarizer(self.summarizer)
        self.blueprint.register_summarizer(self.summarizer)

        # Try to write filenames somewhere
        try:
            dl_train.dataset.write_file_names_to_txt(self.log_dir)
        except AttributeError:
            raise AttributeError(f'dl_train.dataset of type {type(dl_train.dataset)} does not support '
                                 f'write_file_names_to_txt(log_dir)!')

        # superclass setup
        super(MultiscaleTrainer, self).__init__(dl_train, [self.optim], net, sw,
                                                max_epochs=self.config_dl.max_epochs,
                                                log_config=log_config, saver=saver, skip_to_itr=skip_to_itr)

    def _custom_init(self):
        pass

    @staticmethod
    def get_style_from_config(config_p: str):
        components = config_p.split(os.path.sep)
        if 'clf' in components:
            return 'classifier'
        return 'enhancement'

    def modules_to_save(self):
        return {'net': self.blueprint.net,
                'optim': self.optim}

    def _get_dataloaders(self, num_workers, shuffle_train=True):
        print('Cropping to {}'.format(self.config_dl.crop_size))

        ds_train = self.get_ds_train()
        assert shuffle_train, 'shuffle_train=False unsupported'
        dl_train = DataLoader(ds_train, self.config_dl.batchsize_train,
                              sampler=PersistentRandomSampler(data_source=ds_train),
                              num_workers=num_workers)
        print('Created DataLoader [train] {} batches -> {} imgs'.format(
                len(dl_train), self.config_dl.batchsize_train * len(dl_train)))

        ds_val = self._get_ds_val(
                self.config_dl.imgs_dir_val,
                crop=self.config_dl.crop_size_val)  # no cropping, only here for pytest!

        print(f'Created DataSet [val] {len(ds_val)} imgs')

        # for tensorboard: crop to 192, and get first only
        fixed_first_val = self._get_ds_val(
                self.config_dl.imgs_dir_val,
                crop=min(128, self.config_dl.crop_size or 128))[0]
        print(f'Created fixed_first [val]; keys: {fixed_first_val.keys()}')

        return dl_train, ds_val, fixed_first_val

    def get_ds_train(self):
        """
        Dataset must return dicts with {'idx', 'raw'}, where 'raw' is 3HW uint8
        """
        if self.config_dl.is_residual_dataset:
            return get_residual_dataset(self.config_dl.imgs_dir_train, random_transforms=True,
                                        random_scale=self.config_dl.random_scale,
                                        crop_size=self.config_dl.crop_size,
                                        mode='both' if self.style == 'enhancement' else 'diff',
                                        discard_shitty=self.config_dl.discard_shitty_train,
                                        filter_min_size=self.config_dl.train_filter_min_size,
                                        top_only=global_config.get('top_only', None), is_training=True)
        else:
            assert self.style != 'enhancement', 'style == enhancement -> expected residual dataset'

        to_tensor_transform = transforms.Compose(
                [transforms.RandomCrop(self.config_dl.crop_size),
                 transforms.RandomHorizontalFlip(),
                 images_loader.IndexImagesDataset.to_tensor_uint8_transform()])


        if global_config.get('ycbcr', False):
            print('Adding ->YCbCr')
            t = transforms.Lambda(lambda pil_img: pil_img.convert('YCbCr'))
            to_tensor_transform.transforms.insert(-2, t)

        ds_syn = global_config.get('ds_syn', None)
        if ds_syn:
            ds_train = self._get_syn(ds_syn, 30*10000)
        else:
            ds_train = images_loader.IndexImagesDataset(
                    images=cached_listdir_imgs(
                            self.config_dl.imgs_dir_train,
                            min_size=self.config_dl.crop_size,
                            discard_shitty=self.config_dl.discard_shitty_train),
                    to_tensor_transform=to_tensor_transform)
        return ds_train

    def _get_ds_val(self, imgs_dir_val, crop=False):
        # ---
        # ds_syn = global_config.get('ds_syn', None)
        # if ds_syn:
        #     ds_val = self._get_syn(ds_syn, truncate)
        #     return ds_val
        # ---
        if self.config_dl.is_residual_dataset:
            return get_residual_dataset(self.config_dl.imgs_dir_val, random_transforms=False, random_scale=False,
                                        crop_size=crop, mode='both' if self.style == 'enhancement' else 'diff',
                                        discard_shitty=self.config_dl.discard_shitty_val)
        else:
            assert self.style != 'enhancement', 'style == enhancement -> expected residual dataset'

        img_to_tensor_t = shared.get_test_dataset_transform(crop)

        ds = images_loader.IndexImagesDataset(
                images=cached_listdir_imgs(
                        imgs_dir_val,
                        min_size=self.config_dl.crop_size,
                        discard_shitty=True),
                to_tensor_transform=img_to_tensor_t)

        return ds

    def _get_syn(self, ds_syn, num_els):
        if ds_syn.startswith('cb'):
            ds_kinds = {
                'cb': lambda: IndexDataset(make_checkerboard_dataset(crop_size=self.config_dl.crop_size,
                                                                     values=[0, 255],
                                                                     pattern_sizes=[1],
                                                                     num_els=num_els)),
                'cbB': lambda: IndexDataset(make_checkerboard_dataset(crop_size=self.config_dl.crop_size,
                                                                      values=[10, 240],
                                                                      pattern_sizes=[1],
                                                                      num_els=num_els))
            }
            try:
                return ds_kinds[ds_syn]()
            except KeyError:
                raise KeyError(ds_kinds.keys())
        else:  # multi_N_maxPat
            _, n, max_pattern_size = ds_syn.split('_')
            colors = list(np.linspace(10, 240, int(n)))
            print(round(np.log2(int(max_pattern_size))))
            pats = [2 ** p for p in range(int(round(np.log2(int(max_pattern_size)))) + 1)]
            return IndexDataset(make_checkerboard_dataset(crop_size=self.config_dl.crop_size,
                                                          values=colors,
                                                          pattern_sizes=pats,
                                                          num_els=num_els))

    def train_step(self, i, batch, log, log_heavy, load_time_per_batch=None):
        """
        :param i: current step
        :param batch: dict with 'idx', 'raw'
        """
        self.lr_schedule.update(i)
        self.net.zero_grad()

        values = Values('{:.3e}', ' | ')

        with self.time_accumulator.execute():
            # x_n is NormalizedTensor
            # TODO(enh): figure out a cleaner design. for now:
            # x_n is a tuple (raw, lossy) for style == enhancement
            # out is then EnhancementOut
            x_n, bpps, n_sp_pre_pad = self.blueprint.unpack(batch)

            with self.summarizer.maybe_enable(prefix='train', flag=log, global_step=i):
                out = self.blueprint.forward(x_n, bpps)

            with self.summarizer.maybe_enable(prefix='train', flag=log_heavy, global_step=i):
                loss = self.blueprint.losses(out,
                                             num_subpixels_before_pad=n_sp_pre_pad,
                                             base_bpp=bpps.mean())

            total_loss = loss.total_loss
            total_loss.backward()
            self.optim.step()

            # Note: whatever is added here will be shown in TB!
            values['loss'] = total_loss

            if self.style == 'enhancement':
                bpsp = loss.bpsp_base + loss.bpsp_residual
                values['bpsp'] = bpsp

        if not log:
            return

        if load_time_per_batch:
            values['load_time_per_batch'] = load_time_per_batch

        mean_time_per_batch = self.time_accumulator.mean_time_spent()
        imgs_per_second = self.config_dl.batchsize_train / mean_time_per_batch

        print(f'{self.log_date} {i: 6d}: {values.get_str()} // '
              f'{imgs_per_second:.3f} img/s')

        values.write(self.sw, i)

        # log LR
        lrs = list(self.get_lrs())
        assert len(lrs) == 1
        self.sw.add_scalar('train/lr', lrs[0], i)

        if not log_heavy:
            return

        self.blueprint.add_image_summaries(self.sw, out, i, 'train')

    def validation_step(self, i, kind):
        if kind == 'fixed_first':
            with self.summarizer.enable(prefix='val', global_step=i):
                x_n, bpps, n_sp_pre_pad = self.blueprint.unpack_batch_pad(self.fixed_first_val)
                out = self.blueprint.forward(x_n, bpps)
                _ = self.blueprint.losses(out,
                                          num_subpixels_before_pad=n_sp_pre_pad,
                                          base_bpp=bpps.mean())
        elif kind == 'validation_set':
            with timer.execute(f'>>> Running on {len(self.ds_val)} images of validation set [s]'):
                test_id = TestID(self.ds_val.id, i)
                test_results = TestResults()

                for idx, img in enumerate(self.ds_val):
                    filename = self.ds_val.get_filename(idx)
                    x_n, bpps, n_sp_pre_pad = self.blueprint.unpack_batch_pad(img)
                    out = self.blueprint.forward(x_n, bpps)
                    loss = self.blueprint.losses(out,
                                                 num_subpixels_before_pad=n_sp_pre_pad,
                                                 base_bpp=bpps.mean())

                    test_results.set_from_loss(loss, filename)

                _, test_output_cache = multiscale_tester.get_test_log_dir_and_cache(
                        self.log_dir_root, os.path.basename(self.log_dir))
                print(f'VALIDATE  {i: 6d}: {test_results.means_str()}')
                test_output_cache[test_id] = test_results
                for key, value in test_results.means_dict().items():
                    self.sw.add_scalar(f'val_set/{self.ds_val.id}/{key}',
                                       value, i)
        else:
            raise ValueError('Invalid kind', kind)


    @staticmethod
    def _store_training_imgs(dataset: images_loader.IndexImagesDataset, log_dir):
        assert os.path.isdir(log_dir)
        with open(os.path.join(log_dir, 'train_imgs.txt'), 'w') as fout:
            fout.write(f'{len(dataset.files)} images, {dataset.id}\n---\n')
            fout.write('\n'.join(map(os.path.basename, dataset.files)))






class Values(object):
    """
    Stores values during one training step. Essentially a thin wrapper around dict with support to get a nicely
    formatted string and write to a SummaryWriter.
    """
    def __init__(self, fmt_str='{:.3f}', joiner=' / ', prefix='train/'):
        self.fmt_str = fmt_str
        self.joiner = joiner
        self.prefix = prefix
        self.values = {}

    def __setitem__(self, key, value):
        try:
            self.values[key] = value.item()
        except AttributeError:
            self.values[key] = value

    def get_str(self):
        """ :return pretty printed version of all values, using default_fmt_str """
        return self.joiner.join('{} {}'.format(k, self.fmt_str.format(v))
                                for k, v in sorted(self.values.items()))

    def write(self, sw, i, skip=None):
        """ Writes to summary writer `sw`. """
        for k, v in self.values.items():
            if skip and k in skip:
                continue
            sw.add_scalar(self.prefix + k, v, i)


class ValuesAcc(object):
    def __init__(self):
        self.values = None
        self.reset()

    def __setitem__(self, key, value):
        try:
            self.values[key].append(value.item())
        except AttributeError:
            self.values[key].append(value)

    def set_values(self, v: Values):
        for key, values in self.values.items():
            v[key] = np.mean(values)
        return self

    def reset(self):
        self.values = defaultdict(list)
