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

from fjcommon import timer
from torchvision import transforms

from blueprints.classifier_blueprint import ClassifierOut
from dataloaders import images_loader
from dataloaders.classifier_data import ClassifierDataset
from test import multiscale_tester
from test.test_helpers import TestID, TestResults
from train.multiscale_trainer import MultiscaleTrainer, Values, ValuesAcc


class ClassifierTrainer(MultiscaleTrainer):
    def get_ds_train(self):
        t = [transforms.RandomHorizontalFlip(),
             images_loader.IndexImagesDataset.to_tensor_uint8_transform()]
        if self.config_dl.crop_size:
            print('Random cropping train to', self.config_dl.crop_size)
            t.insert(0, transforms.RandomCrop(self.config_dl.crop_size))
        to_tensor_transform = transforms.Compose(t)
        return ClassifierDataset(self.config_dl.imgs_dir_train,
                                 self.config_dl.optimal_qs_csv_train,
                                 to_tensor_transform)

    def _get_ds_val(self, imgs_dir_val, crop=False):
        # NOTE: no cropping validation!
        to_tensor_transform = transforms.Compose(
                [images_loader.IndexImagesDataset.to_tensor_uint8_transform()])
        return ClassifierDataset(self.config_dl.imgs_dir_val,
                                 self.config_dl.optimal_qs_csv_val,
                                 to_tensor_transform)

    def train_step(self, i, batch, log, log_heavy, load_time_per_batch=None):
        self.lr_schedule.update(i)
        self.net.zero_grad()
        self.optim.zero_grad()

        # if self.overfit_batch is None:
        #     print('***OVERFITTING')
        #     self.overfit_batch = batch
        # batch = self.overfit_batch

        with self.time_accumulator.execute():
            x_n, q = self.blueprint.unpack(batch)

            with self.summarizer.maybe_enable(prefix='train', flag=log, global_step=i):
                out: ClassifierOut = self.blueprint.forward(x_n)

            with self.summarizer.maybe_enable(prefix='train', flag=log_heavy, global_step=i):
                loss = self.blueprint.loss(out.q_logits, q)

            loss.backward()
            self.optim.step()

            # Note: whatever is added here will be shown in TB!
            self.values_acc['loss'] = loss
            self.values_acc['acc'] = self.blueprint.get_accuracy(out.q_logits, q)

        if not log:
            return

        values = Values('{:.3e}', ' | ')

        if load_time_per_batch:
            values['load_time_per_batch'] = load_time_per_batch

        self.values_acc.set_values(values).reset()

        mean_time_per_batch = self.time_accumulator.mean_time_spent()
        imgs_per_second = self.config_dl.batchsize_train / mean_time_per_batch

        print(f'{self.log_date} {i: 6d}: {values.get_str()} // '
              f'{imgs_per_second:.3f} img/s')

        values.write(self.sw, i)

        # log LR
        lrs = list(self.get_lrs())
        assert len(lrs) == 1
        self.sw.add_scalar('train/lr', lrs[0], i)

    def _custom_init(self):
        self.values_acc = ValuesAcc()

        self.overfit_batch = None



    def validation_step(self, i, kind):
        if kind == 'fixed_first':
            with self.summarizer.enable(prefix='val', global_step=i):
                x_n, q = self.blueprint.unpack_batch_pad(self.fixed_first_val)
                out: ClassifierOut = self.blueprint.forward(x_n)
        elif kind == 'validation_set':
            with timer.execute(f'>>> Running on {len(self.ds_val)} images of validation set [s]'):
                test_id = TestID(self.ds_val.id, i)
                test_results = TestResults()

                for idx, img in enumerate(self.ds_val):
                    filename = self.ds_val.get_filename(idx)
                    x_n, q = self.blueprint.unpack_batch_pad(img)
                    out: ClassifierOut = self.blueprint.forward(x_n)
                    loss = self.blueprint.loss(out.q_logits, q)
                    test_results.set_from_loss(loss, filename)
                    test_results.set(filename, 'acc', self.blueprint.get_accuracy(out.q_logits, q))

                _, test_output_cache = multiscale_tester.get_test_log_dir_and_cache(
                        self.log_dir_root, os.path.basename(self.log_dir))
                test_output_cache[test_id] = test_results
                print(f'VALIDATE  {i: 6d}: {test_results.means_str()}')
                for key, value in test_results.means_dict().items():
                    self.sw.add_scalar(f'val_set/{self.ds_val.id}/{key}',
                                       value, i)
        else:
            raise ValueError('Invalid kind', kind)

