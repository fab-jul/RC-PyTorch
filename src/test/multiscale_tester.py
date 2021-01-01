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
import os
import auto_crop
import shutil
import time
from functools import total_ordering
from typing import List, Dict
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from fjcommon import functools_ext as ft, config_parser, timer, no_op
from fjcommon.assertions import assert_exc

import gather_tests
import pytorch_ext as pe
from blueprints.classifier_blueprint import ClassifierBlueprint
from blueprints.enhancement_blueprint import EnhancementBlueprint, QStrategy, EnhancementLoss, EnhancementOut
from helpers.pad import pad
from helpers.summaries import new_bottleneck_summary
from dataloaders.compressed_images_loader import MetaResidualDataset
from helpers import paths, saver, logdir_helpers
from helpers.config_checker import DEFAULT_CONFIG_DIR
from helpers.global_config import global_config
from helpers.quantized_tensor import NormalizedTensor, SymbolTensor
from helpers.testset import Testset
from test import cuda_timer
from test.image_saver import ImageSaver
from test.test_helpers import TestID, TestResults, TestOutputCache, get_test_log_dir_root, CropMeans, LossOutMeta, \
    QHistory
from vis.image_summaries import to_image

# used for the Shared RGB basline
_DEFAULT_RECURSIVE_FOR_RGB = 3

_FILE_EXT = '.l3c'

_CLEAN_CACHE_PERIODICALLY = int(os.environ.get('CUDA_CLEAN_CACHE', '0')) == 1

_LIVE_HISTORY = int(os.environ.get('LIVE_HISTORY', '0')) == 1
if _LIVE_HISTORY:
    print('*** LIVE HISTORY')


GLOBAL_CONFIG_TEST_PREFIX = 'test.'


class MultiscaleTesterInitException(Exception):
    pass


@total_ordering
class CheckerboardTestset(object):
    def __init__(self):
        self.id = 'checkerboard'

    def filter_filenames(self, _):
        pass

    # to enable sorting
    def __lt__(self, other):
        return False


class EncodeError(Exception):
    pass


class DecodeError(Exception):
    pass


_KNOWN_DATASETS = {
    'CDIV2K_valid_HR_crop4_bpg_q12_None_dS=False_DIV2K_valid_HR_crop4_None_dS=False': 'DIV2K_CROP',
    'Cmobile_valid_crop4_bpg_q12_None_dS=False_mobile_valid_crop4_None_dS=False': 'CLIC_mobile',
    'Cprofessional_valid_crop4_bpg_q12_None_dS=False_professional_valid_crop4_None_dS=False': 'CLIC_pro',
    'Cval_oi_500_bpg_q12_None_dS=False_val_oi_500_None_dS=False': 'OI_VAL'
}


# Uniquely identifies a test run of some experiment
# dataset_id comes from Testset.id, which is 'FOLDERNAME_NUMIMGS'

def _parse_recursive_flag(recursive, config_ms):
    if not config_ms.rgb_bicubic_baseline:
        return 0
    if recursive == 'auto':
        if config_ms.rgb_bicubic_baseline and config_ms.num_scales == 1:  # RGB - shared
            return _DEFAULT_RECURSIVE_FOR_RGB
    try:
        return int(recursive)
    except ValueError:
        return 0


def _clean_cuda_cache(i):
    if i % 25 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()


# TODO:
# this is actually not "default" but the only supported, as it gets stripped from log dir
_DEFAULT_PREFIXES = ['ms', 'dl']


class CropLossCombinator(auto_crop.CropLossCombinator):
    def __init__(self, base_bpp):
        self._base_bpp = base_bpp
        super(CropLossCombinator, self).__init__()

    def get_combined_loss(self) -> EnhancementLoss:
        # TODO: support MultiscaleLoss
        return EnhancementLoss(
            total_loss=None, bpsp_residual=self.get_bpsp(), bpsp_base=self._base_bpp / 3)


def get_test_log_dir_and_cache(log_dir, experiment_basename, reset_entire_cache=False):
    test_log_dir_root = get_test_log_dir_root(log_dir)
    # E.g. test_log_dir/0311_1057 cr oi_012
    test_log_dir = os.path.join(test_log_dir_root, experiment_basename)
    if reset_entire_cache and os.path.isdir(test_log_dir):
        print(f'Removing test_log_dir={test_log_dir}...')
        time.sleep(1)
        shutil.rmtree(test_log_dir)
    os.makedirs(test_log_dir, exist_ok=True)
    test_output_cache = TestOutputCache(test_log_dir)
    return test_log_dir, test_output_cache


class MultiscaleTester(object):
    def __init__(self, log_date, flags, restore_itr, l3c=False, configs_dir=None, style='multiscale',
                 filter_ckpts_at=None):
        """
        :param flags:
            log_dir
            img
            filter_filenames
            max_imgs_per_folder
            # out_dir
            crop
            recursive
            sample
            write_to_files
            write_means
            compare_theory
            time_report
            overwrite_cache
            save_imgs
            clf_p
        """
        self.flags = flags
        self.style = style

        self.blueprint_cls = {'enhancement': EnhancementBlueprint,
                              'classifier': ClassifierBlueprint}[style]

        if style == 'classifier':
            self._test = self._classifier_test

        for prefix in ('',): # 'AWS'):
            try:
                config_ps, self.experiment_dir = MultiscaleTester.get_configs_experiment_dir(
                        os.path.join(self.flags.log_dir, prefix), log_date, update_global_config=True,
                        configs_dir=configs_dir or DEFAULT_CONFIG_DIR)
                break
            except ValueError as e:
                print(f'*** Caught error: {e}')
        else:  # no-break
            raise MultiscaleTesterInitException(f'Cannot find {log_date} in {self.flags.log_dir}')

        self.log_date = logdir_helpers.log_date_from_log_dir(self.experiment_dir)
        # we do not need the data loading config
        (self.config_ms, config_dl), rel_paths = ft.unzip(map(config_parser.parse, config_ps))
        self.config_ms.is_residual = style in ('enhancement',)

        # Update global_config given config.global_config
        try:
            global_config_from_config = self.config_ms.global_config
        except AttributeError:
            global_config_from_config = None
        global_config.add_from_str_without_overwriting(global_config_from_config)

        global_config.update_config(self.config_ms)

        if self.style == 'enhancement':
            EnhancementBlueprint.read_evenly_spaced_bins(config_dl)

        print('Using global_config:', global_config)

        # TODO(enh)
        blueprint = self.blueprint_cls(self.config_ms, is_testing=True)
        blueprint.set_eval()
        self.blueprint = blueprint

        if self.flags.clf_p:
            print('*** Setting classifier...')
            assert style == 'enhancement'
            blueprint: EnhancementBlueprint = self.blueprint
            blueprint.add_classifier(self.flags.clf_p)

        if self.flags.tau_optimization:
            print('*** Enabling TAU...')
            assert style == 'enhancement'
            blueprint: EnhancementBlueprint = self.blueprint
            blueprint.enable_tau_optimization()

        if self.flags.qstrategy:
            print(f'*** Enabling QSTRATEGY={self.flags.qstrategy}...')
            assert style == 'enhancement'
            blueprint: EnhancementBlueprint = self.blueprint
            blueprint.qstrategy = QStrategy[self.flags.qstrategy]

        self.test_log_dir, self.test_output_cache = get_test_log_dir_and_cache(
                self.flags.log_dir,
                os.path.basename(self.experiment_dir.rstrip(os.path.sep)),
                self.flags.reset_entire_cache)

        restore_itr = self._parse_restore_itr(restore_itr, self.test_output_cache)
        self.restorer = saver.Restorer(paths.get_ckpts_dir(self.experiment_dir))

        for try_nr in range(2):
            try:
                self.restore_itr, ckpt_p = self.restorer.get_ckpt_for_itr(restore_itr, before_time=filter_ckpts_at)
                self.restorer.restore({'net': self.blueprint.net}, ckpt_p, strict=True)
                break
            except EOFError as e:  # probably because of ckpt getting deleted while reading
                print('*** Caught', e)
                time.sleep(5)
        else:  # no-break, not loaded
            raise MultiscaleTesterInitException('Could not restore!')

        self.times = cuda_timer.StackTimeLogger() if self.flags.write_to_files else None

        # Import only if needed, as it imports torchac
        if self.flags.write_to_files or l3c:
            if self.flags.write_to_files:
                kwargs = {'times': self.times, 'compare_with_theory': self.flags.compare_theory}
            else:
                kwargs = {'times': cuda_timer.StackTimeLogger(), 'compare_with_theory': False}
            print('Creating Bitcoding')
            if style == 'enhancement':
                from bitcoding.bitcoding_enh import Bitcoding
                self.bc = Bitcoding(self.blueprint, **kwargs)
            else:
                from bitcoding.bitcoding import Bitcoding
                self.bc = Bitcoding(self.blueprint, **kwargs)

    @staticmethod
    def _parse_restore_itr(restore_itr, test_output_cache: TestOutputCache):
        if restore_itr == 'latest_cached':
            return test_output_cache.latest_cached()
        if 'k' in restore_itr:
            return int(restore_itr[:-1]) * 1000
        return int(restore_itr)

    @staticmethod
    def get_configs_experiment_dir(log_dir, log_date, update_global_config,
                                   prefixes=None, configs_dir=DEFAULT_CONFIG_DIR):
        if not prefixes:
            prefixes = _DEFAULT_PREFIXES
        experiment_dir = paths.get_experiment_dir(log_dir, log_date)
        log_dir_comps = logdir_helpers.parse_log_dir(
                experiment_dir, configs_dir, prefixes, append_ext='.cf')
        config_ps = log_dir_comps.config_paths
        if update_global_config:
            global_config.reset(keep=GLOBAL_CONFIG_TEST_PREFIX)
            global_config.add_from_flag(log_dir_comps.postfix)
        return config_ps, experiment_dir

    @ft.return_list
    def _get_results(self, datasets, fn):
        """ Collect results into dict """
        results = [fn(ds) for ds in datasets]
        for testset, result in zip(datasets, results):
            if self.flags.gather:
                print('Gathering', testset, result)
                # TODO: only works for meta
                gather_tests.Gatherer(self.flags, verbose=False, dataset=(testset, testset.name)).write(
                        result, self.log_date, self.restore_itr)

            d = result.means_dict()
            d.update({'testset': _KNOWN_DATASETS.get(testset.id, testset.id),
                      'exp': self.log_date,
                      'itr': self.restore_itr})
            yield d

    # Testing ----------------------------------------------------------------------

    def test_all(self, datasets) -> List[Dict[str, object]]:
        assert not self.flags.write_to_files  # no results generated
        return self._get_results(datasets, self.test)

    def test(self, ds) -> TestResults:
        """
        :param ds: A DataSet subclass with an .id flag
        :return:
        """
        assert hasattr(ds, 'id'), type(ds)

        test_id = TestID(ds.id, self.restore_itr)
        return_cache = (not self.flags.overwrite_cache and
                        not self.flags.write_to_files and
                        test_id in self.test_output_cache)

        if return_cache:
            print(f'*** Found cached: {test_id}')


            return self.test_output_cache[test_id]

        print('Testing {}'.format(ds))
        # ds = self.get_test_dataset(testset)
        with torch.no_grad():
            with timer.execute('>>> entire testing loop'):
                results = self._test(ds)
            assert results is not None

        self.test_output_cache[test_id] = results
        return results

    def _classifier_test(self, ds) -> TestResults:
        pass

    # ~160s on 500 images
    def _test(self, ds) -> TestResults:
        test_results = TestResults()

        log = ''

        # p_out = join(self.out_dir_clean, f'{fn}_{i}.png')
        if self.style == 'enhancement' and ds.get_filename(0).endswith('_0'):
            print('*** Crop dataset detected')
            crop_means = CropMeans()
        else:
            crop_means = no_op.NoOp

        is_meta = isinstance(ds, MetaResidualDataset)
        if is_meta:
            print('*** MetaResidualDataset', ds.id)
            is_meta = ds.starting_q
        # Counts how often which Q is optimal
        qs = defaultdict(int)
        # TODO: update NoOp to work for if foo and len(foo) and other builtins
        qs[None] = 1  # bc w/o MetaResidualDataset, qs[None] will be updated. But we want len(qs) > 1
        modulo_op = self.flags.modulo_op  # None or int in {1,2,3}
        if modulo_op:
            assert 1<=modulo_op<=3, modulo_op
        out_dir = 'q_histories' + (f'_{modulo_op}' if modulo_op else '')
        q_history = (QHistory(self.log_date, TestID(ds.id, self.restore_itr), out_dir=out_dir)
                     if is_meta else
                     no_op.NoOp)

        if _LIVE_HISTORY:
            existing_files = set(q_history.read())
            ds.set_skip(existing_files, modulo_op)
            print(f'*** Storing at {q_history.out_path()}')

        # If we sample, we store the result with a ImageSaver
        if self.flags.sample:
            image_saver = ImageSaver(os.path.join(self.flags.sample, self.log_date),
                                     merge=global_config.get('test.merge', False),
                                     trim=3)
            print('Will store samples in {}.'.format(image_saver.out_dir))
        else:
            image_saver = None

        _clean_cuda_cache(i=0)
        loss_out_meta_mem = LossOutMeta.create_memory(is_meta, self.blueprint)
        for i, img in enumerate(ds):
            # `img` can be different things:
            # - for L3C: TODO
            # - for single image enhancement:  dict with keys raw, compressed, bpps
            #   in this case, ds is a ResidualDataset
            # - for meta image enhancement: dict with { Q -> raw, compressed, bpps } }
            #   in this case, ds is a MetaResidualDataset

            if img is None:
                continue
            if modulo_op and i % 3 != (modulo_op - 1):
                print('***Skipping', i)
                continue

            filename = ds.get_filename(i)

            if _CLEAN_CACHE_PERIODICALLY:
                _clean_cuda_cache(i)

            loss_out_meta = LossOutMeta(is_meta=is_meta, blueprint=self.blueprint)

            # Note: bpps are the bpps of the BPG images.
            for (raw, compressed, bpps), q in loss_out_meta.unroll_unpack(img):
                combinator = CropLossCombinator(bpps.mean() if bpps else 0.)

                for crop_idx, (raw_crop, compressed_crop) in enumerate(zip(
                        auto_crop.iter_crops(raw), auto_crop.iter_crops(compressed))):
                    assert raw_crop.shape == compressed_crop.shape

                    # We have to pad images not divisible by (2 ** num_scales), because we downsample num_scales-times.
                    # To get the correct bpsp, we have to use, num_subpixels_before_pad,
                    #   see `get_loss` in multiscale_blueprint.py
                    x_n_crop, _, n_sp_pre_pad = self.blueprint.pad_pack(raw_crop, compressed_crop, bpps)
                    out = self.blueprint.forward(x_n_crop, bpps)  # Note: bpps only used for conditional IN!

                    # NOTE: if --tau_optimization is given, this updates `out.network_out` in-place!
                    #       That means that then, --sample will also use the tau...
                    loss_out = self.blueprint.losses(
                        out,
                        num_subpixels_before_pad=n_sp_pre_pad,
                        # We set bpsp on combined_loss_out. This function here only uses it to put it into the tuple.
                        base_bpp=0.)

                    if not isinstance(loss_out, EnhancementLoss):  # TODO.
                        raise NotImplementedError('Loss not supported: {}'.format(loss_out))

                    combinator.add(loss_out.bpsp_residual, num_subpixels_crop=n_sp_pre_pad)

                    if self.flags.sample:
                        _, x_c = x_n_crop
                        self._sample(x_c, loss_out, out, image_saver, f'{filename}_{crop_idx}')

                combined_loss_out = combinator.get_combined_loss()
                loss_out_meta.append(combined_loss_out, q)

            # extract minimum
            loss_out, q_min = loss_out_meta.get_min(memory=loss_out_meta_mem)
            qs[q_min] += 1
            q_history[filename] = q_min
            test_results.set_from_loss(loss_out, filename)
            if q_min:
                test_results.set(filename, 'Q', q_min)
            crop_means.add(filename, loss_out_meta.n_sp_pre_pad, loss_out)

            # if self.flags.write_means:
            #     predicted_mean = out.get_mean_img()
            #     os.makedirs(self.flags.write_means, exist_ok=True)
            #     p_out = os.path.join(self.flags.write_means, filename + '.npy')
            #     self._save_diff(inp=x_n.to_sym().get(),
            #                     otp=NormalizedTensor(predicted_mean, L=256).to_sym().get(),
            #                     p_out=p_out)
            #
            #     bpp_zs = 3 * sum(loss_out.all_bpsps[1:])
            #     bpp_file_p = os.path.join(self.flags.write_means, 'bpps.txt')
            #     with open(bpp_file_p, 'a') as f:
            #         f.write(f'{bpp_zs}  # {filename}\n')


            log = f'{self.log_date}: {filename} ({i: 10d}): {test_results.means_str()}'
            # self.blueprint.losses.print_tau_optimization_summary()

            if len(qs) > 1:  # not just None as a key
                qs_str = ' '.join(f'{q}: {count}' for q, count in sorted(qs.items(), key=lambda qc: -qc[1])
                                  if q is not None)
                log += f' // {qs_str}    '
            log += loss_out_meta.memory_to_str(loss_out_meta_mem, joiner=' ; ')
            _print(log, oneline=True)
            if _LIVE_HISTORY:
                q_history.write_out(verbose=False)
        _print('FINAL:' + log, oneline=True, final=True)

        crop_means.write_to_results(test_results)
        q_history.write_out(verbose=True)


        return test_results

    # Write to files ---------------------------------------------------------------

    def write_to_files(self, datasets) -> List[Dict[str, object]]:
        """ Pendant to test_all """
        assert self.config_ms.is_residual
        assert self.flags.write_to_files
        return self._get_results(datasets, self._write_to_files)

    def _write_to_files(self, ds) -> TestResults:
        out_dir = self.flags.write_to_files
        os.makedirs(out_dir, exist_ok=True)

        log = ''

        test_results = TestResults()

        for i, img in enumerate(ds):
            # if self.blueprint.clf is not None:
            #
            #     print(img.keys())
            #     exit(1)
            #     pass
            raw_p = ds.get_raw_p(i)
            print('***', raw_p)

            with self.times.skip(i == 0):
                filename = os.path.splitext(os.path.basename(raw_p))[0]
                out_p = os.path.join(out_dir, filename + _FILE_EXT)
                actual_bpsp = self._write_to_file(raw_p, out_p)
                print(actual_bpsp)
                test_results.set(filename, 'bpsp', actual_bpsp)
                print('*' * 80)

            log = f'{self.log_date}: {filename} ({i: 10d}): {test_results.means_str()}'
            _print(log, oneline=True)
        _print(log, oneline=True, final=True)

        return test_results

    def _write_to_file(self, raw_p, out_p):
        """
        :param img: 1CHW, long
        :param out_p: string
        :return: info string
        """
        if os.path.isfile(out_p):
            os.remove(out_p)

        with self.times.run('=== bc.encode'):
            encode_out = self.bc.encode(raw_p, pout=out_p)

        with self.times.run('=== bc.decode'):
            img_o = self.bc.decode(pin=out_p)

        # print(encode_out.img.dtype, encode_out.img.min(), encode_out.img.max())
        # print(img_o.dtype, img_o.min(), img_o.max())
        pe.assert_equal(encode_out.img, img_o)

        print('\n'.join(self.times.get_last_strs()))
        if self.flags.time_report:
            with open(self.flags.time_report, 'w') as f:
                f.write('Average times:\n')
                f.write('\n'.join(self.times.get_mean_strs()))

        if encode_out.info_str:
            print(encode_out.info_str)

        return encode_out.actual_bpsp

    def encode(self, img_p, pout, overwrite=False):
        pout_dir = os.path.dirname(os.path.abspath(pout))
        assert_exc(os.path.isdir(pout_dir), f'pout directory ({pout_dir}) does not exists!', EncodeError)
        if overwrite and os.path.isfile(pout):
            print(f'Removing {pout}...')
            os.remove(pout)
        assert_exc(not os.path.isfile(pout), f'{pout} exists. Consider --overwrite', EncodeError)

        img = self._read_img(img_p)
        img = img.to(pe.DEVICE)

        self.bc.encode(img, pout=pout)
        print('---\nSaved:', pout)

    def decode(self, pin, png_out_p):
        """
        Decode L3C-encoded file at `pin` to a PNG at `png_out_p`.
        """
        pout_dir = os.path.dirname(os.path.abspath(png_out_p))
        assert_exc(os.path.isdir(pout_dir), f'png_out_p directory ({pout_dir}) does not exists!', DecodeError)
        assert_exc(png_out_p.endswith('.png'), f'png_out_p must end in .png, got {png_out_p}', DecodeError)

        decoded = self.bc.decode(pin)

        self._write_img(decoded, png_out_p)
        print(f'---\nDecoded: {png_out_p}')

    def _read_img(self, img_p):
        img = np.array(Image.open(img_p)).transpose(2, 0, 1)  # Turn into CHW
        C, H, W = img.shape
        # check number of channels
        if C == 4:
            print('*** WARN: Will discard 4th (alpha) channel.')
            img = img[:3, ...]
        elif C != 3:
            raise EncodeError(f'Image has {C} channels, expected 3 or 4.')
        # Convert to 1CHW torch tensor
        img = torch.from_numpy(img).unsqueeze(0).long()
        # Check padding
        padding = self.blueprint.get_padding_fac()
        if H % padding != 0 or W % padding != 0:
            print(f'*** WARN: image shape ({H}X{W}) not divisible by {padding}. Will pad...')
            img,_  = pad(img, fac=padding, mode='constant')
        return img

    @staticmethod
    def _write_img(decoded, png_out_p):
        """
        :param decoded: 1CHW tensor
        :param png_out_p: str
        """
        # TODO: should undo padding
        assert decoded.shape[0] == 1 and decoded.shape[1] == 3, decoded.shape
        img = pe.tensor_to_np(decoded.squeeze(0))  # CHW
        img = img.transpose(1, 2, 0).astype(np.uint8)  # Make HW3
        img = Image.fromarray(img)
        img.save(png_out_p)

    # TODO: note that this only works for enhancement mode now
    def _sample(self, x_l: NormalizedTensor, loss_out: EnhancementLoss, out: EnhancementOut, image_saver, save_prefix):
        # Make sure folder does not already contain samples for this file.
        if image_saver.file_starting_with_exists(save_prefix):
            # raise FileExistsError('Previous sample outputs found in {}. Please remove.'.format(
            #         image_saver.out_dir))
            print('Previous samples found, skipping:', save_prefix)
            return

        bpg_bpsp = loss_out.bpsp_base
        total_bpsp = bpg_bpsp + loss_out.bpsp_residual

        def _to_np(t: SymbolTensor):
            return t.get().to(torch.int16).detach().cpu().numpy()

        def _to_img(t: np.ndarray, mi, ma):
            t = torch.from_numpy(t).float()
            t_n = normalize(t, mi, ma).mul(255.).round().long()
            # find most frequent value and set it to white (255)
            vs, cs = torch.unique(t_n, return_counts=True)
            vs = [v for c, v in sorted(zip(cs, vs), reverse=True)]
            most_frequent = vs[0]
            t_n[t_n==most_frequent] = 255
            return t_n

        # fn_res = f'{save_prefix}_{total_bpsp:.3f}'
        # fn_res = save_prefix
        # Store res as npy and PNG
        res_gt, fn_gt = image_saver.save_res(_to_np(out.res_sym), (save_prefix, '_res_gt.npy'))
        # if res is not None:  # once all crops were saved!
        #     assert '.npy' in fn
        #     image_saver.save_img(_to_img(torch.from_numpy(res).float()), fn.replace('.npy', '.png'))
        ## image_saver.save_img(x_l.to_sym().get(), (save_prefix, '_bpg.png'))
        # Store ground truth for comparison
        # note: x_r is sym
        image_saver.save_img(out.x_r.get(), (save_prefix, '_raw.png'))

        for i in range(2):
            sampled = self.blueprint.sample_forward(out.network_out).to_sym()
            res_sampled, fn_sampled = image_saver.save_res(_to_np(sampled), (save_prefix, f'_res_sample{i}.npy'))
            if i == 0 and res_sampled is not None:
                assert '.npy' in fn_gt
                assert '.npy' in fn_sampled
                mi = min(np.amin(res_gt), np.amin(res_sampled))
                ma = max(np.amax(res_gt), np.amax(res_sampled))
                print(mi, ma)
                image_saver.save_img(_to_img(res_gt, mi, ma), fn_gt.replace('.npy', '.png'))
                image_saver.save_img(_to_img(res_sampled, mi, ma), fn_sampled.replace('.npy', '.png'))
                break

    def _save_bn(self, i, z):
        out_dir = os.path.join(self.test_log_dir, 'bns')
        os.makedirs(out_dir, exist_ok=True)

        out_p = os.path.join(out_dir, f'bn_{i:010d}.png')
        print(out_p)

        img = new_bottleneck_summary(z)
        img = Image.fromarray(to_image(img))
        img.save(out_p)
        return

        zks = []

        for k in range(K):
            zk = z.get()[0, k, ...]  # long, 0...L
            zk = zk.float().div(z.L).mul(255).round().to(torch.uint8)
            zks.append(zk)

            Image.fromarray(zk.detach().cpu().numpy()).save(out_p)

    def _save_in_out(self, fn, x_n, means):
        d = self.log_date
        os.makedirs(d, exist_ok=True)
        self._save_img(x_n, os.path.join(d, f'{fn}_i.png'))
        self._save_img(NormalizedTensor(means, L=256), os.path.join(d, f'{fn}_o.png'))

    @staticmethod
    def _save_img(t: NormalizedTensor, out_p):
        print('\nSaving', out_p)
        img = t.to_sym().get().to(torch.uint8)[0, ...].permute(1, 2, 0).detach().cpu().numpy()
        Image.fromarray(img).save(out_p)

    @staticmethod
    def _save_diff(inp, otp, p_out):
        print(f'*** Saving diff {p_out}')
        assert inp.shape == otp.shape
        assert len(inp.shape) == 4 and inp.shape[1] == 3

        def _to_chw_np(t):
            return t[0, ...].permute(1, 2, 0).cpu().detach().numpy().astype(np.int16)

        diff = _to_chw_np(inp) - _to_chw_np(otp)
        np.save(p_out, diff)


def _print(s, oneline, final=False):
    if oneline:
        if not final:
            print('\r' + s, end='')
        else:
            print('\n' + s)
    else:
        print(s)


def _url_of_img(img_name, testset,
                public_root='/home/mentzerf/public_html/datasets',
                url_root='http://people.ee.ethz.ch/~mentzerf/datasets'):
    testset_public = os.path.join(public_root, testset.name)
    if not os.path.isdir(testset_public):
        print('Copying to', testset_public)
        os.makedirs(testset_public, 0o755)
        os.chmod(testset_public, 0o755)
        for pout in testset.copy_to(testset_public):
            os.chmod(pout, 0o755)
    try:
        img_out_name = next(bn for bn in os.listdir(testset_public) if img_name in bn)
    except StopIteration:
        raise ValueError('Not found:', img_name)

    return os.path.join(url_root, testset.name, img_out_name)


def _write_to_csv(test_results: TestResults, testset: Testset, out_file_p: str, other_codec: dict = None):
    if len(testset) <= 1:
        raise ValueError()
    with open(out_file_p, 'w') as fout:
        res, header = test_results.per_image_results()
        for k in other_codec.keys():
            header += [f'bpp_{k}', f'psnr_{k}']
        fout.write(','.join(['image'] + header) + '\n')
        for img_name, results in res.items():
            url = _url_of_img(img_name, testset)
            row = [f'=IMAGE("{url}")'] + list(map(str, results))
            if other_codec:
                # other_codec is codec -> img_name -> (bpp, psnr)
                for k, per_img in other_codec.items():
                    assert img_name in per_img, img_name
                    row += list(map(str, per_img[img_name]))
                    print('updated row', row)
            fout.write(','.join(row) + '\n')
        fout.write('\n')
    print('Created', out_file_p)


def normalize(t, mi=None, ma=None):
    if not mi:
        mi = t.min()
    if not ma:
        ma = t.max()
    return t.add(-mi).div(ma - mi + 1e-5)
