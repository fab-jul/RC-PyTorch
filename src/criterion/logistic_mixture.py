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

This class is based on the TensorFlow code of PixelCNN++:
    https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
In contrast to that code, we predict mixture weights pi for each channel, i.e., mixture weights are "non-shared".
Also, x_min, x_max and L are parameters, and we implement a function to get the CDF of a channel.

# ------
# Naming
# ------

Note that we use the following names through the code, following the code PixelCNN++:
    - x: targets, e.g., the RGB image for scale 0
    - l: for the output of the network;
      In Fig. 2 in our paper, l is the final output, denoted with p(z^(s-1) | f^(s)), i.e., it contains the parameters
      for the mixture weights.
"""

from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from fjcommon import functools_ext as ft

import pytorch_ext as pe
import vis.grid
import vis.summarizable_module
from helpers.global_config import global_config
from helpers.quantized_tensor import NormalizedTensor
# Note that for RGB, we predict the parameters mu, sigma, pi and lambda. Since RGB has C==3 channels, it so happens that
# the total number of channels needed to predict the 4 parameters is 4 * C * K (for K mixtures, see final paragraphs of
# Section 3.4 in the paper). Note that for an input of, e.g., C == 4 channels, we would need 3 * C * K + 6 * K channels
# to predict all parameters. To understand this, see Eq. (7) in the paper, where it can be seen that for \tilde \mu_4,
# we would need 3 lambdas.
# We do not implement this case here, since it would complicate the code unnecessarily.
from modules.prob_clf import NetworkOutput
from vis.summarizable_module import normalize_to_0_1

_NUM_PARAMS_RGB = 4  # mu, sigma, pi, lambda
_NUM_PARAMS_OTHER = 3  # mu, sigma, pi

_MAX_K_FOR_VIS = 10

_X_LOWER_BOUND = -0.999
_X_UPPER_BOUND = 0.999


CDFOut = namedtuple('CDFOut', ['logit_probs_c_sm',
                               'means_c',
                               'log_scales_c',
                               'K',
                               'targets'])


def non_shared_get_Kp(K, C):
    """ Get Kp=number of channels to predict. See note where we define _NUM_PARAMS_RGB above """
    if C == 3:  # finest scale
        return _NUM_PARAMS_RGB * C * K
    else:
        return _NUM_PARAMS_OTHER * C * K


def non_shared_get_K(Kp, C):
    """ Inverse of non_shared_get_Kp, get back K=number of mixtures """
    if C == 3:
        return Kp // (_NUM_PARAMS_RGB * C)
    else:
        return Kp // (_NUM_PARAMS_OTHER * C)


# --------------------------------------------------------------------------------


def test_plot():
    rgb_scale = True

    if rgb_scale:
        dmll = DiscretizedMixLogisticLoss(rgb_scale=True, L=256)
    else:
        dmll = DiscretizedMixLogisticLoss(rgb_scale=False, L=25)

    from dataloaders import checkerboard
    case = checkerboard._generate_pattern([10, 240], 8, 1)
    case = torch.from_numpy(case).unsqueeze(0).to(torch.float32)

    print(case.shape)

    N, C, H, W = case.shape
    K = 1

    p = _NUM_PARAMS_RGB

    l = torch.arange(N * C * K * p * H * W, dtype=torch.float32)
    l = l.reshape(N, p, C, K, H, W)
    # Pi
    l[0, 0, :, 0, ...] = 1
    # means
    l[0, 1, :, 0, ...] = -0.5
    l[0, 1, :, 0, 0, 1] = 0.5
    l[0, 1, :, 0, 1, 0] = 0.5
    # sigma
    l[0, 2, :, 0, ...] = -7
    # lambda
    l[0, 3, :, 0, ...] = 0

    l = NetworkOutput(means=l[:, 1, ...],
                      sigmas=l[:, 2, ...],
                      pis=l[:, 0, ...],
                      lambdas=l[:, 3,...])

    # dmll._alpha = -1


    import matplotlib.pyplot as plt
    from helpers.quantized_tensor import SymbolTensor
    case_n = SymbolTensor(case, 256).to_norm()
    dmll.plot(case_n, l, plt)

    plt.show()

    bc = dmll.forward(case_n, l, 0)
    print(bc.flatten())


def test_x_range():
    num_per_series = 4
    for H in (2, 10, 128):
        hits = np.zeros((H, H))
        print('H ==', H)
        x_range, y_range = _get_series_range(H, num_per_series), _get_series_range(H, num_per_series)
        # for offset in (0, 1):
        for x in x_range:
            for y in y_range:

                hits[x, y] += 1
        print(hits)


class DiscretizedMixLogisticLoss(vis.summarizable_module.SummarizableModule):
    def __init__(self, rgb_scale: bool, L):
        """
        :param rgb_scale: Whether this is the loss for the RGB scale. In that case,
            use_coeffs=True
            _num_params=_NUM_PARAMS_RGB == 4, since we predict coefficients lambda. See note above.
        :param L: number of symbols
        """
        super(DiscretizedMixLogisticLoss, self).__init__()

        self.rgb_scale = rgb_scale
        self.L = L

        self._means_oracle = global_config.get('means_oracle', None)
        if self._means_oracle:
            print('*** Means oracle,', self._means_oracle)

        self._self_auto_reg = global_config.get('s_autoreg', False)

        # Adapted bounds for our case.
        self.bin_width = 2 / (L-1)

        # Lp = L+1
        self.targets = torch.linspace(-1 - self.bin_width / 2,
                                      1  + self.bin_width / 2,
                                      self.L + 1, dtype=torch.float32, device=pe.DEVICE)

        self.min_sigma = global_config.get('minsigma', -9.)

        self._extra_repr = (f'DMLL:'
                            f'L={self.L}, '
                            f'bin_width={self.bin_width}, min_sigma={self.min_sigma}')

        self._alpha = 1

    def extra_repr(self):
        return self._extra_repr

    # @staticmethod
    # def to_per_pixel(entropy, C):
    #     N, H, W = entropy.shape
    #     return entropy.sum() / (N*C*H*W)  # NHW -> scalar

    def cdf_step_non_shared(self, l: NetworkOutput, c_cur, targets, x_c=None) -> CDFOut:
        """
        :param l:
        :param c_cur:
        :param targets: because we don't know the range a priori
        :param x_c:  x *up to* c, NCHW still
        :return:
        """
        # NKHW         NKHW     NKHW
        logit_probs_c, means_c, log_scales_c, K = self._extract_non_shared_c(c_cur, l, x_c)
        logit_probs_c_softmax = F.softmax(logit_probs_c, dim=1)  # NKHW, pi_k
        return CDFOut(logit_probs_c_softmax, means_c, log_scales_c, K, targets)

    def sample(self, l: NetworkOutput) -> NormalizedTensor:
        return self._non_shared_sample(l)

    def plot(self, x: NormalizedTensor, l: NetworkOutput, plt,
             x_range=None, y_range=None, num_per_series=2):
        """
        :param x: NCHW
        :param l:  NKpHW
        :param plt:
        :return:
        """
        # Extract ---
        #  NCKHW      NCKHW  NCKHW
        x, logit_pis, means, log_scales = self._extract_non_shared(x, l)

        _, axs = self._make_subplot(plt, x.t.shape)
        self._plot_cdf(x, logit_pis, means, log_scales, axs,
                       x_range, y_range, num_per_series)
        # plt.show()

    @staticmethod
    def _make_subplot(plt, x_shape):
        fig, axs = plt.subplots(x_shape[1], 2, sharex=False)
        return fig, axs

    def _plot_cdf(self, x: NormalizedTensor, logit_pis, means, log_scales, axs,
                  x_range=None, y_range=None, num_per_series=2):
        """
        :param x: NC1HW
        :param logit_pis:
        :param means:
        :param log_scales:
        :param axs: where to plot to
        :param num_per_series:
        :return:
        """

        _, _, _, H, W = x.get().shape
        if not x_range:
            assert num_per_series > 0
            x_range = _get_series_range(H, num_per_series)
        if not y_range:
            assert num_per_series > 0
            y_range = _get_series_range(W, num_per_series)

        # NCKHW -> 1CK44
        cut = lambda t_: t_.detach()[:1, :, :, slice(*x_range), slice(*y_range)]
        logit_pis, means, log_scales = map(cut, (logit_pis, means, log_scales))

        # Get first element in batch
        cdf = self._get_cdf(logit_pis, means, log_scales).detach()[0, ...].cpu().numpy()

        C, H, W, Lp = cdf.shape
        # CHW
        sym = x.to_sym().get().detach()[0, :, 0, slice(*x_range), slice(*y_range)].cpu().numpy()

        blow_up = 8
        x_vis = x.get()[0, :, 0, slice(*x_range), slice(*y_range)].permute(1, 2, 0)
        x_vis = normalize_to_0_1(x_vis).mul(255).round().to(torch.uint8)
        x_vis = x_vis.detach().cpu().numpy().repeat(blow_up, axis=0).repeat(blow_up, axis=1)

        # offset = 0
        # if self.L > 256:
        #     offset = 256
        # targets = np.arange(Lp) - offset
        # print(len(targets))

        gts = set()
        for c, (ax_a, ax) in enumerate(axs):
            print(ax)
            ax_a.imshow(x_vis[..., c], cmap='gray')
            for x in range(H):
                for y in range(W):
                    cdf_xy = cdf[c, x, y, 1:] - cdf[c, x, y, :-1]
                    p = ax.plot(np.arange(Lp-1), cdf_xy, linestyle='-', linewidth=0.5)
                    ax.set_ylim(-0.1, 1.1)
                    gt = sym[c, x, y]
                    gts.add(gt)
                    ax.axvline(gt, color=p[-1].get_color(), linestyle='--', linewidth=0.5)
        for _, ax in axs:
            ax.set_xlim(min(gts) - 5, max(gts) + 5)


    def _get_cdf(self, logit_probs, means, log_scales):
        """
        :param logit_probs: NCKHW
        :param means: Updated w.r.t. some x! NCKHW
        :param log_scales: NCKHW
        :return:
        """
        # NCKHW1
        inv_stdv = torch.exp(-log_scales).unsqueeze(-1)
        # NCKHWL'
        centered_targets = (self.targets - means.unsqueeze(-1))
        # NCKHWL'
        cdf_k = centered_targets.mul(inv_stdv).sigmoid()
        # NCKHW1, pi_k
        logit_probs_softmax = F.softmax(logit_probs, dim=2).unsqueeze(-1)
        # NCHWL'
        cdf = cdf_k.mul(logit_probs_softmax).sum(2)
        return cdf

    def forward(self, x: NormalizedTensor, l: NetworkOutput, scale=0):
        """
        :param x: labels, i.e., NCHW, float
        :param l: predicted distribution, i.e., NKpHW, see above
        :return: log-likelihood, as NHW if shared, NCHW if non_shared pis
        """
        # Extract ---
        #  NCKHW      NCKHW  NCKHW
        x, logit_pis, means, log_scales = self._extract_non_shared(x, l)

        x_raw = x.get()

        # visualize pi, means, variances
        # note: val is fixed_val
        #       train is normal train but log_heavy only!
        self.summarizer.register_images(
                'val', {f'dmll/{scale}/c{c}': lambda c=c: _visualize_params(logit_pis, means, log_scales, c)
                          for c in range(x_raw.shape[1])})
        self.summarizer.register_figures(
                'val',
                {f'dmll/{scale}/cdf': lambda axs_: self._plot_cdf(x, logit_pis, means, log_scales, axs_)},
                fig_creator=lambda plt_: self._make_subplot(plt_, x_raw.shape))

        if scale == 0:
            self.summarizer.register_scalars(
                    'train',
                    {f'dmll/{scale}/scales/{c}/mean': lambda c=c: torch.exp(-log_scales[:, c, ...]).mean()
                     for c in range(x_raw.shape[1])})

        bitcost = self.forward_raw(x_raw, log_scales, logit_pis, means)

        # notebook_dict.notebook_dict['bitcost'] = bitcost.detach()

        # TODO: inconsistent naming
        self.summarizer.register_images(
                'val', {f'dmll/bitcost/{scale}': lambda: _visualize_bitcost(bitcost[0, ...])})

        return bitcost

    def forward_raw(self, x_raw, log_scales, logit_pis, means):
        centered_x = x_raw - means  # NCKHW
        # Calc P = cdf_delta
        # all of the following is NCKHW
        inv_stdv = torch.exp(-log_scales)  # <= exp(7), is exp(-sigma), inverse std. deviation, i.e., sigma'
        plus_in = inv_stdv * (centered_x + self.bin_width / 2)  # sigma' * (x - mu + 0.5)
        # exp(log_scales) == sigma -> small sigma <=> high certainty
        cdf_plus = torch.sigmoid(plus_in)  # S(sigma' * (x - mu + 1/255))
        min_in = inv_stdv * (centered_x - self.bin_width / 2)  # sigma' * (x - mu - 1/255)
        cdf_min = torch.sigmoid(min_in)  # S(sigma' * (x - mu - 1/255)) == 1 / (1 + exp(sigma' * (x - mu - 1/255))
        # the following two follow from the definition of the logistic distribution
        log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0
        log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255
        # NCKHW, P^k(c)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases, essentially log_cdf_plus + log_one_minus_cdf_min
        mid_in = inv_stdv * centered_x  # sigma' * x
        # log probability in the center of the bin, to be used in extreme cases
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
        # NOTE: the original code has another condition here:
        #   tf.where(cdf_delta > 1e-5,
        #            tf.log(tf.maximum(cdf_delta, 1e-12)),
        #            log_pdf_mid - np.log(127.5)
        #            )
        # which handles the extremly low porbability case.
        #
        # so, we have the following if, where I put in the X_UPPER_BOUND and X_LOWER_BOUND values for RGB
        # if x < 0.001:                         cond_C
        #       log_cdf_plus                    out_C
        # elif x > 254.999:                     cond_B
        #       log_one_minus_cdf_min           out_B
        # else:
        #       log(cdf_delta)                  out_A
        out_A = torch.log(torch.clamp(cdf_delta, min=1e-12))
        out_A_pdf = log_pdf_mid + np.log(self.bin_width)
        # self.summarizer.register_scalars(
        #         'val',
        #         {f'dmll/{scale}/pdf_cdf': lambda: (out_A - out_A_pdf).mean()})
        cond_A = (cdf_delta > 1e-5).float()
        cond_B = (x_raw > _X_UPPER_BOUND).float()
        cond_C = (x_raw < _X_LOWER_BOUND).float()
        out_A = (cond_A * out_A) + \
                (1. - cond_A) * out_A_pdf
        out_B = (cond_B * log_one_minus_cdf_min) + \
                (1. - cond_B) * out_A
        log_probs = (cond_C * log_cdf_plus) + \
                    (1. - cond_C) * out_B  # NCKHW, =log(P^k(c))
        # combine with pi, NCKHW, (-inf, 0]
        log_probs_weighted = log_probs.add(
                log_softmax(logit_pis, dim=2))  # (-inf, 0]
        # TODO:
        # for some reason, this somehow can become negative in some elements???
        # final log(P), NCHW
        bitcost = -log_sum_exp(log_probs_weighted, dim=2)  # NCHW
        return bitcost

    def _extract_non_shared(self, x: NormalizedTensor, l: NetworkOutput):
        """
        :param x: targets, NCHW
        :param l: output of net, NKpHW, see above
        :return:
            x NC1HW,
            logit_probs NCKHW (probabilites of scales, i.e., \pi_k)
            means NCKHW,
            log_scales NCKHW (variances),
            K (number of mixtures)
        """
        x_raw = x.get()

        N, C, H, W = x_raw.shape

        logit_probs = l.pis  # NCKHW
        means = l.means  # NCKHW
        log_scales = torch.clamp(l.sigmas, min=self.min_sigma)  # NCKHW, is >= -MIN_SIGMA

        x_raw = x_raw.reshape(N, C, 1, H, W)

        if l.lambdas is not None:
            assert C == 3  # Coefficients only supported for C==3, see note where we define _NUM_PARAMS_RGB
            coeffs = torch.tanh(l.lambdas)  # NCKHW, basically coeffs_g_r, coeffs_b_r, coeffs_b_g
            means_r, means_g, means_b = means[:, 0, ...], means[:, 1, ...], means[:, 2, ...]  # each NKHW
            coeffs_g_r,  coeffs_b_r, coeffs_b_g = coeffs[:, 0, ...], coeffs[:, 1, ...], coeffs[:, 2, ...]  # each NKHW
            x_reg = means if self._self_auto_reg else x_raw
            means = torch.stack(
                    (means_r,
                     means_g + coeffs_g_r * x_reg[:, 0, ...],
                     means_b + coeffs_b_r * x_reg[:, 0, ...] + coeffs_b_g * x_reg[:, 1, ...]), dim=1)  # NCKHW again

            if self._means_oracle:
                mse_pre = F.mse_loss(means, x_raw)
                diff = (x_raw - means).detach()
                means += self._means_oracle * diff
                mse_cur = F.mse_loss(means, x_raw)
                self.summarizer.register_scalars(
                        'val', {f'dmll/0/oracle_mse_impact': lambda: mse_cur - mse_pre})

            # TODO: will not work for RGB baseline
            self.summarizer.register_scalars(
                    'train', {f'dmll/0/coeffs_{c}': lambda c=c: coeffs[:, c, ...].detach().mean()
                            for c in range(C)})

        x = NormalizedTensor(x_raw, x.L)
        return x, logit_probs, means, log_scales

    # TODO: Normalized
    def _extract_non_shared_c(self, c, l: NetworkOutput, x_raw=None):
        """
        :param x_raw: NCHW
        Same as _extract_non_shared but only for c-th channel, used to get CDF
        """
        N, C, K, H, W = l.means.shape

        assert c < C, f'{c} >= {C}'

        logit_probs_c = l.pis[:, c, ...]  # NKHW
        means_c = l.means[:, c, ...]  # NKHW
        log_scales_c = torch.clamp(l.sigmas[:, c, ...], min=self.min_sigma)  # NKHW, is >= -7

        if l.lambdas is not None and c != 0:
            assert x_raw is not None
            x_raw = x_raw.reshape(N, C, 1, H, W)
            unscaled_coeffs = l.lambdas  # NCKHW, coeffs_g_r, coeffs_b_r, coeffs_b_g
            if c == 1:
                coeffs_g_r = torch.tanh(unscaled_coeffs[:, 0, ...])  # NKHW
                # NKHW     NKHW         N1HW
                means_c += coeffs_g_r * x_raw[:, 0, ...]
            elif c == 2:
                coeffs_b_r = torch.tanh(unscaled_coeffs[:, 1, ...])  # NKHW
                coeffs_b_g = torch.tanh(unscaled_coeffs[:, 2, ...])  # NKHW
                # also NKHW
                means_c += coeffs_b_r * x_raw[:, 0, ...] + coeffs_b_g * x_raw[:, 1, ...]

        #      NKHW           NKHW     NKHW
        return logit_probs_c, means_c, log_scales_c, K

    def _non_shared_sample(self, l: NetworkOutput) -> NormalizedTensor:
        """ sample from model """
        logit_probs = l.pis  # NCKHW
        N, C, K, H, W = logit_probs.shape

        # sample mixture indicator from softmax
        u = torch.zeros_like(logit_probs).uniform_(1e-5, 1. - 1e-5)  # NCKHW
        sel = torch.argmax(
                logit_probs - torch.log(-torch.log(u)),  # gumbel sampling
                dim=2)  # argmax over K, results in NCHW, specifies for each c: which of the K mixtures to take
        assert sel.shape == (N, C, H, W), (sel.shape, (N, C, H, W))

        sel = sel.unsqueeze(2)  # NC1HW

        means = torch.gather(l.means, 2, sel).squeeze(2)
        log_scales = torch.clamp(torch.gather(l.sigmas, 2, sel).squeeze(2), min=self.min_sigma)

        # sample from the resulting logistic, which now has essentially 1 mixture component only.
        # We use inverse transform sampling. i.e. X~logistic; generate u ~ Unfirom; x = CDF^-1(u),
        #  where CDF^-1 for the logistic is CDF^-1(y) = \mu + \sigma * log(y / (1-y))
        u = torch.zeros_like(means).uniform_(1e-5, 1. - 1e-5)  # NCHW
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))  # NCHW

        if l.lambdas is not None:
            assert C == 3

            clamp = lambda x_: torch.clamp(x_, -1., 1.)

            # Be careful about coefficients! We need to use the correct selection mask, namely the one for the G and
            #  B channels, as we update the G and B means! Doing torch.gather(coeffs, 2, sel) would be completly
            #  wrong.
            coeffs = torch.tanh(l.lambdas)
            sel_g, sel_b = sel[:, 1, ...], sel[:, 2, ...]
            coeffs_g_r = torch.gather(coeffs[:, 0, ...], 1, sel_g).squeeze(1)
            coeffs_b_r = torch.gather(coeffs[:, 1, ...], 1, sel_b).squeeze(1)
            coeffs_b_g = torch.gather(coeffs[:, 2, ...], 1, sel_b).squeeze(1)

            # Note: In theory, we should go step by step over the channels and update means with previously sampled
            # xs. But because of the math above (x = means + ...), we can just update the means here and it's all good.
            x0 = clamp(x[:, 0, ...], )
            x1 = clamp(x[:, 1, ...] + coeffs_g_r * x0)
            x2 = clamp(x[:, 2, ...] + coeffs_b_r * x0 + coeffs_b_g * x1)
            x = torch.stack((x0, x1, x2), dim=1)

        return NormalizedTensor(x, self.L, centered=True)


def log_prob_from_logits(logit_probs):
    """ numerically stable log_softmax implementation that prevents overflow """
    # logit_probs is NKHW
    m, _ = torch.max(logit_probs, dim=1, keepdim=True)
    return logit_probs - m - torch.log(torch.sum(torch.exp(logit_probs - m), dim=1, keepdim=True))


# TODO: replace with pytorch internal in 1.0, there is a bug in 0.4.1
def log_softmax(logit_probs, dim):
    """ numerically stable log_softmax implementation that prevents overflow """
    m, _ = torch.max(logit_probs, dim=dim, keepdim=True)
    return logit_probs - m - torch.log(torch.sum(torch.exp(logit_probs - m), dim=dim, keepdim=True))


def log_sum_exp(log_probs, dim):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    m, _        = torch.max(log_probs, dim=dim)
    m_keep, _   = torch.max(log_probs, dim=dim, keepdim=True)
    # == m + torch.log(torch.sum(torch.exp(log_probs - m_keep), dim=dim))
    return log_probs.sub_(m_keep).exp_().sum(dim=dim).log_().add(m)


def _visualize_params(logits_pis, means, log_scales, channel):
    """
    :param logits_pis:  NCKHW
    :param means: NCKHW
    :param log_scales: NCKHW
    :param channel: int
    :return:
    """
    assert logits_pis.shape == means.shape == log_scales.shape
    logits_pis = logits_pis[0, channel, ...].detach()
    means = means[0, channel, ...].detach()
    log_scales = log_scales[0, channel, ...].detach()

    pis = torch.softmax(logits_pis, dim=0)  # Kdim==0 -> KHW

    mixtures = ft.lconcat(
            zip(_iter_Kdim_normalized(pis, normalize=False),
                _iter_Kdim_normalized(means),
                _iter_Kdim_normalized(log_scales)))
    grid = vis.grid.prep_for_grid(mixtures)
    img = torchvision.utils.make_grid(grid, nrow=3)
    return img


def _visualize_bitcost(logP):
    """
    :param logP: CHW
    :return: img
    """
    C, _, _ = logP.shape
    logP = logP.detach().clamp(min=0)
    # normalize entire thing so that scale is relative. min should remain 0
    logP = logP.div(logP.max() + 1e-5)
    channels = [logP[c, ...].unsqueeze(0) for c in range(C)]
    return torchvision.utils.make_grid(channels, nrow=3)


def _iter_Kdim_normalized(t, normalize=True):
    """ normalizes t, then iterates over Kdim (1st dimension) """
    K = t.shape[0]

    if normalize:
        lo, hi = float(t.min()), float(t.max())
        t = t.clamp(min=lo, max=hi).add_(-lo).div_(hi - lo + 1e-5)

    for k in range(min(_MAX_K_FOR_VIS, K)):
        yield t[k, ...]  # HW


def _get_series_range(max_val, num_per_series):
    start = max(0, (max_val - num_per_series) // 2)
    end = min(max_val, start + num_per_series)
    return start, end
