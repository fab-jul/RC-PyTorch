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

Grid

Say you have K C H W
you want a N/N grid
so you fold to K C N N H//N W//N
and you have taus K C N N 1 1

cool


"""
import collections
import math
import os
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from fjcommon.no_op import NoOp
from torch import nn
from torch.nn import functional as F

from criterion.logistic_mixture import DiscretizedMixLogisticLoss
from helpers.global_config import global_config
from helpers.quantized_tensor import NormalizedTensor
from modules import prob_clf
from test import cuda_timer

VERBOSE_TAU = int(os.environ.get('VERBOSE_TAU', 0))

printv = print if VERBOSE_TAU else NoOp


def _safe_mean(v):
    if v:
        return np.mean(v)
    return None


class _list(object):
    def __init__(self):
        self._content = []

    def append(self, x):
        self._content.append(x)

    def mean(self):
        if self._content:
            return np.mean(self._content)
        return None


@dataclass
class _Summary:
    num_fails: int = 0
    params: dict = None

    gains = _list()
    times = _list()
    diffs = _list()
    losses = _list()  # List of lists

    def add_time(self, t):
        self.times.append(t)

    def add_gain(self, gain):
        self.gains.append(gain)

    def add_diff(self, d):
        self.diffs.append(d)

    @property
    def mean_gain(self):
        return self.gains.mean()

    @property
    def mean_time(self):
        return self.times.mean()

    def add_param(self, name, param):
        if not self.params:
            self.params = {}
        param_summary = torch.flatten(param.detach().cpu())
        param_summary = torch.unique(param_summary)
        self.params[name] = param_summary

    def __str__(self):
        return (f'Summary('
                f'num_fails={self.num_fails}, '
                f'gains={self.gains.mean()}, '
                f'time={self.times.mean()}, '
                f'params={self.get_params_str()})')

    def get_params_str(self):
        if not self.params:
            return ''
        return ','.join(f'{k}={v}' for k, v in sorted(self.params.items())).replace('\n', '')


class TauOptimizationHelper(object):
    def __init__(self,
                 loss_dmol_rgb: DiscretizedMixLogisticLoss):
        self.loss_dmol_rgb = loss_dmol_rgb

        self._plot_loss = global_config.get('test.plot_loss', False)

        self._info = global_config.get('test.info', '')
        self._mode = global_config.get('test.mode', 'direct')  # direct, channel

        self._rand_init = global_config.get('test.rand_init', False)
        self._grid = global_config.get('test.grid', 1)
        self._full = global_config.get('test.full', False)
        self._num_iter = global_config.get('test.num_iter', 50)
        self._early_stop = global_config.get('test.early_stop', False)
        self._ignore_overhead = global_config.get('test.ignore_overhead', False)
        self._subsampling = global_config.get('test.subsampling', 4)
        self._optim_cls = global_config.get('test.optim', 'SGD')
        self._lr = global_config.get('test.lr', 9e-2)
        self._optim_params = global_config.get_as_dict(
            'test.optim_params', 'dict(momentum=0.9)' if self._optim_cls != 'Adam' else 'dict()')

        self._summary = _Summary()

    def save_plot(self, dataset_name, out_dir='tau_optim_plots'):
        title = f'{self._optim_cls}_{self._optim_params}_{self._lr:.3e}_{self._subsampling}' \
                f'_mode={self._mode}_itr={self._num_iter}_early={self._early_stop}'

        import matplotlib as mpl
        mpl.use('Agg')  # No display
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.title(title)

        all_losses = self._summary.losses._content
        for losses in all_losses[:50]:
            plt.plot(range(len(losses)), losses)

        os.makedirs(out_dir, exist_ok=True)
        p_out = os.path.join(out_dir, f'{title}.pdf')
        print(f'Saving {p_out}...')
        plt.savefig(p_out, bbox_inches='tight', pad_inches=0)

        plt.figure(figsize=(6, 6))
        biggest = sorted(all_losses, key=lambda l: max(l) - min(l), reverse=True)[0]
        plt.plot(range(len(biggest)), biggest)
        p_out = os.path.join(out_dir, f'{title}_biggest.pdf')
        plt.savefig(p_out, bbox_inches='tight', pad_inches=0)

    @staticmethod
    def _conv2d(param, weights, **kwargs):
        B, C, K, H, W = param.shape
        return F.conv2d(param.reshape(B, C * K, H, W), weights, **kwargs).reshape(B, C, K, H, W)

    def _get_modified_network_out(self, network_out: prob_clf.NetworkOutput, taus):
        if self._mode == 'direct':
            taus = taus['sigma']
            # Grid upscaling
            if self._grid > 1:
                _, _, _, H, W = network_out.sigmas.shape

                print('taus:', taus[0, 0, ...].flatten())  # Show some taus!

                # We overscale and then truncate to accomodate any H, W.
                upscaling_H, upscaling_W = math.ceil(H / self._grid), math.ceil(W / self._grid)
                taus = taus.repeat_interleave(upscaling_H, 2).repeat_interleave(upscaling_W, 3)
                taus = taus[..., :H, :W]  # Truncate.
                assert taus.shape[-2:] == (H, W), (taus.shape, network_out.sigmas.shape)
            sigmas = network_out.sigmas * taus
        elif self._mode == 'channels':
            taus = taus['sigma']
            sigmas = self._conv2d(network_out.sigmas, taus)
        elif self._mode == 'channels2':
            U, V = taus['sigma_U'], taus['sigma_V']
            bias = 0  # taus['sigma_b'].unsqueeze(-1).unsqueeze(-1)
            weights = torch.matmul(U, V).unsqueeze(-1).unsqueeze(-1)
            sigmas = self._conv2d(network_out.sigmas, weights) + bias + network_out.sigmas
        elif self._mode == 'channels_nl_scaling':
            U, V = taus['sigma_U'], taus['sigma_V']
            weights = torch.matmul(U, V).unsqueeze(-1).unsqueeze(-1)
            sigmas = (torch.tanh(self._conv2d(network_out.sigmas, weights)) + \
                     network_out.sigmas) * taus['sigma_scaling']
        elif self._mode == 'channels_spatial':
            U, V = taus['sigma_U'], taus['sigma_V']
            f = 5
            _, C, K, _, _ = network_out.sigmas.shape
            Ci = C * K
            weights = torch.zeros(Ci, Ci, f, f, device='cuda')
            weights += torch.matmul(U, V).unsqueeze(-1).unsqueeze(-1)
            for i in range(Ci):
                weights[i, i, ...] += taus['filter']
            sigmas = self._conv2d(network_out.sigmas, weights, padding=f//2)
        elif self._mode == 'spatial':
            taus = taus['sigma']
            _, C, K, _, _ = network_out.sigmas.shape
            f = 5
            Ci = C * K
            weight = torch.zeros(Ci, Ci, f, f, device='cuda')
            for i in range(Ci):
                weight[i, i, ...] += taus
            sigmas = self._conv2d(network_out.sigmas, weight, padding=f//2)
        else:
            raise ValueError

        return prob_clf.copy_network_output(network_out, sigmas=sigmas)

    def print_summary(self):
        print(self._summary)


    def get_google_sheets_row(self):
        params = [
            self._info,
            self._mode,
            self._grid,
            self._rand_init,
            self._num_iter,
            self._early_stop,
            self._ignore_overhead,
            self._subsampling,
            self._optim_cls,
            str(self._optim_params),
            self._lr]
        summary = [
            self._summary.num_fails,
            self._summary.mean_gain,
            self._summary.get_params_str(),
            self._summary.mean_time,
            self._summary.diffs.mean()]
        return [*params, *summary]

    def _get_taus(self, network_out_ss):
        _, C, K, H, W = network_out_ss.sigmas.shape

        if self._mode == 'direct':
            if self._full:
                initializer = torch.ones(C, K, H, W, requires_grad=True, device='cuda')
            else:
                initializer = torch.ones(C, K, self._grid, self._grid, requires_grad=True, device='cuda')
            if self._rand_init:
                noise = torch.rand_like(initializer).sub(0.5).mul(.2)
                print('** adding noise', noise.flatten())
                initializer = initializer + noise
        elif self._mode == 'channels':
            # A 1x1 filter
            initializer = torch.normal(torch.eye(C * K), std=0.01).to('cuda').reshape(
                C * K, C * K, 1, 1)
        elif self._mode == 'channels2':
            initializer = {
                'sigma_U': torch.normal(torch.zeros(C * K, 2), std=0.02).to('cuda'),
                'sigma_V': torch.normal(torch.zeros(2, C * K), std=0.02).to('cuda'),
                # 'sigma_b': torch.normal(torch.zeros(C, K), std=0.02).to('cuda'),
            }
        elif self._mode == 'channels_nl_scaling':
            initializer = {
                'sigma_U': torch.normal(torch.zeros(C * K, 2), std=0.02).to('cuda'),
                'sigma_V': torch.normal(torch.zeros(2, C * K), std=0.02).to('cuda'),
                'sigma_scaling': torch.normal(torch.ones(C, K, 1, 1), std=0.02).to('cuda'),
                # 'sigma_b': torch.normal(torch.zeros(C, K), std=0.02).to('cuda'),
            }
        elif self._mode == 'channels_spatial':
            f_i = torch.zeros(5, 5, device='cuda')
            f_i[2, 2] = 1
            f_i = torch.normal(f_i, std=0.01).to('cuda')

            initializer = {
                'sigma_U': torch.normal(torch.zeros(C * K, 2), std=0.02).to('cuda'),
                'sigma_V': torch.normal(torch.zeros(2, C * K), std=0.02).to('cuda'),
                'filter': f_i
            }
        elif self._mode == 'spatial':
            # A 5x5 filters
            # initializer = torch.normal(torch.eye(5), std=0.01).to('cuda')
            initializer = torch.zeros(5, 5, device='cuda')
            initializer[2, 2] = 1
            initializer = torch.normal(initializer, std=0.01).to('cuda')
        else:
            raise ValueError(f'Invalid mode: {self._mode}')

        if not isinstance(initializer, dict):
            initializer = {'sigma': initializer}

        taus = {k: nn.Parameter(v, requires_grad=True) for k, v in initializer.items()}
        return taus

    def optimize(self,
                 res: NormalizedTensor,
                 network_out: prob_clf.NetworkOutput) -> Tuple:
        if VERBOSE_TAU:
            cuda_timer.sync()
            start = time.time()

        with torch.enable_grad():
            network_out_ss = prob_clf.map_over(
                network_out, lambda f: f.detach()[..., ::self._subsampling, ::self._subsampling])

            # Subsample residual.
            res_ss = NormalizedTensor(
                res.t.detach()[..., ::self._subsampling, ::self._subsampling],
                res.L,
                res.centered)

            taus = self._get_taus(network_out_ss)

            optim_cls = {
                'SGD': torch.optim.SGD,
                'Adam': torch.optim.Adam,
                'RMSprop': torch.optim.RMSprop}[self._optim_cls]

            optim = optim_cls(taus.values(), lr=self._lr, **self._optim_params)
            tau_overhead_bytes = (sum(np.prod(tau.shape) * 4 for tau in taus.values())  # 4 bytes per float32.
                                  if not self._ignore_overhead
                                  else 0)

            loss_prev = None
            diffs = collections.deque(maxlen=5)
            initial = None

            losses = [] if self._plot_loss else None

            for i in range(self._num_iter):
                for tau in taus.values():
                    if tau.grad is not None:
                        tau.grad.detach_()
                        tau.grad.zero_()
                        tau.grad = None

                # forward pass
                network_out_ss_tau = self._get_modified_network_out(network_out_ss, taus)
                nll = self.loss_dmol_rgb.forward(res_ss, network_out_ss_tau)

                loss = nll.mean()
                if self._plot_loss:
                    losses.append(loss.item())

                if initial is None:
                    initial = loss.item()

                if loss_prev is not None:
                    diff = loss_prev - loss.item()
                    printv(f'\ritr {i}: {loss.item():.3f} '
                           f'// {diff:.3e} '
                           f'// gain: {initial - loss.item():.5f}', end='', flush=True)
                    diffs.append(abs(diff))
                    if self._early_stop and (len(diffs) >= 5 and np.mean(diffs) < 1e-4):
                        printv('\ndone after', i)
                        break
                loss_prev = loss.item()
                loss.backward()
                optim.step()
                optim.zero_grad()

            if losses:
                print('\n\n***\n', losses, '\n***\n\n')
                self._summary.losses.append(losses)

        if VERBOSE_TAU:
            cuda_timer.sync()
            # noinspection PyUnboundLocalVariable
            diff = time.time() - start
            self._summary.add_time(diff)
            printv(f'time for tau optim: {diff}')

        self._summary.add_diff(np.mean(diffs))

        # Note that this is not the real gain, since it's sub-sampled.
        # Note that this does not take overhead into account!
        final_subsampled_gain = initial - loss.item()
        if final_subsampled_gain < 0:
            printv('*** Was for nothing...')
            self._summary.num_fails += 1
            nll = self.loss_dmol_rgb.forward(res, network_out)
            return nll, None
        else:
            self._summary.add_gain(final_subsampled_gain)
            for k, tau in taus.items():
                self._summary.add_param(f'taus_{k}', tau)
            nll = self.loss_dmol_rgb.forward(
                res, self._get_modified_network_out(network_out, taus))
            # nll is without the overhead of tau
            return nll, tau_overhead_bytes

