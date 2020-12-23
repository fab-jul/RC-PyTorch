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
from collections import namedtuple

import torch
from torch import nn

import pytorch_ext as pe
# from criterion.logistic_mixture import non_shared_get_Kp
from helpers.global_config import global_config
from modules import edsr, act
from modules.gdn import GDN
import torch.nn.functional as F

conv = pe.default_conv

# Each is a NCKHW tensor, or None
NetworkOutput = namedtuple('NetworkOutput', ['means', 'sigmas', 'pis', 'lambdas'])


def copy_network_output(network_out: NetworkOutput, means=None, sigmas=None):
    return NetworkOutput(means if means is not None else network_out.means,
                         sigmas if sigmas is not None else network_out.sigmas,
                         network_out.pis,
                         network_out.lambdas)


def map_over(network_out: NetworkOutput, fn):
    return NetworkOutput(
        fn(network_out.means),
        fn(network_out.sigmas),
        fn(network_out.pis),
        fn(network_out.lambdas),
    )


def extract_mean_image_with_logits(l: NetworkOutput):
    means = _maybe_auto_reg(l)

    if l.means.shape[2] > 1:  # K > 1 -> must average with pis
        logit_pis_sm = F.softmax(l.pis, dim=2)  # NCKHW, pi_k
        means_scaled = means * logit_pis_sm
        means = means_scaled.sum(2)
    else:
        means = means[:, :, 0, ...]
    return means.clamp(-1, 1)


def _maybe_auto_reg(l: NetworkOutput):
    if not global_config.get('s_autoreg', False):
        return l.means

    coeffs = torch.tanh(l.lambdas)  # NCKHW, basically coeffs_g_r, coeffs_b_r, coeffs_b_g
    means_r, means_g, means_b = l.means[:, 0, ...], l.means[:, 1, ...], l.means[:, 2, ...]  # each NKHW
    coeffs_g_r, coeffs_b_r, coeffs_b_g = coeffs[:, 0, ...], coeffs[:, 1, ...], coeffs[:, 2, ...]  # each NKHW
    x_reg = l.means
    return torch.stack(
            (means_r,
             means_g + coeffs_g_r * x_reg[:, 0, ...],
             means_b + coeffs_b_r * x_reg[:, 0, ...] + coeffs_b_g * x_reg[:, 1, ...]), dim=1)  # NCKHW again


# Each is a bool
RequiredOutputs = namedtuple('RequiredOutputs', ['means', 'sigmas', 'pis', 'lambdas'])


def _parse_outputs_flag(outputs):
    """
    can be
    - means,vars,pis,coeffs
    - means,vars,pis
    etc
    :return tuple, whether means, vars, pis, coeffs should be outputted
    """
    valid = ('means', 'vars', 'pis', 'coeffs')
    given = set(outputs.split(','))
    assert all(g in valid for g in given), given
    return RequiredOutputs(*[v in given for v in valid])


def _layers_or_none(flag, layers_factory):
    if flag:
        return layers_factory()
    return lambda x: None


class ProbClfTail(nn.Module):
    @staticmethod
    def get_cout(config, C=3):
        return config.prob.K * C

    @staticmethod
    def from_config(config, C=3, tail_networks=None):
        return ProbClfTail(config.Cf, C, config.prob.K, _parse_outputs_flag(config.prob.rgb_outputs),
                           tail_networks=tail_networks)

    def __init__(self, Cf, C, K, outputs: RequiredOutputs, tail_networks=None):
        super(ProbClfTail, self).__init__()

        self.C, self.K = C, K

        Cout = C * K

        if not tail_networks:
            tail_networks = {}
        if 'default' not in tail_networks:
            tail_networks['default'] = lambda: conv(Cf, Cout, 1)

        invalid_keys = set(tail_networks.keys()) - {'default', 'means', 'sigmas', 'pis', 'lambdas'}
        assert len(invalid_keys) == 0, invalid_keys

        def get_network(kind):
            return tail_networks.get(kind, tail_networks['default'])

        self.means      = _layers_or_none(outputs.means, get_network('means'))
        self.sigmas     = _layers_or_none(outputs.sigmas, get_network('sigmas'))
        self.pis        = _layers_or_none(outputs.pis, get_network('pis'))
        self.lambdas    = _layers_or_none(outputs.lambdas, get_network('lambdas'))

        # self._saver = pe.FeatureMapSaver('tmp/new_age_feat_maps', 'final_feat')

    def _reshape(self, x):
        if x is None:
            return None
        N, _, H, W = x.shape
        return x.reshape(N, self.C, self.K, H, W)

    def forward(self, x) -> NetworkOutput:
        # x = self._saver(x)
        return NetworkOutput(self._reshape(self.means(x)),
                             self._reshape(self.sigmas(x)),
                             self._reshape(self.pis(x)),
                             self._reshape(self.lambdas(x)))


class DeepProbabilityClassifier(nn.Module):
    def __init__(self, config_ms, scale, C=3):
        super(DeepProbabilityClassifier, self).__init__()

        Cf = config_ms.Cf
        kernel_size = 3

        m_body = [
            edsr.ResBlock(conv, Cf, kernel_size, act=act.make(Cf, inverse=True),
                          res_scale=global_config.get('res_scale', 1))
            for _ in range(3)
        ]
        m_body.append(conv(Cf, Cf, kernel_size))

        self.body = nn.Sequential(*m_body)

        K = config_ms.prob.K

        # For RGB, generate the outputs specified by config_ms.prob.rgb_outputs
        # otherwise, generate means, sigmas, pis
        tail_outputs = (_parse_outputs_flag(config_ms.prob.rgb_outputs) if scale == 0
                        else RequiredOutputs(True, True, True, lambdas=False))

        self.tail = ProbClfTail(Cf, C, K, outputs=tail_outputs)


    def forward(self, x) -> NetworkOutput:
        x = self.body(x) + x
        x = self.tail(x)
        return x


class Final(nn.Module):
    def __init__(self, config_ms, C=3, filter_size=3):
        super(Final, self).__init__()

        raise NotImplementedError

        self.C = C
        self.K = config_ms.prob.K

        self.scales_conv = nn.Conv2d(self.C * self.K, self.C * self.K, filter_size,
                                     padding=filter_size//2, bias=global_config.get('initbias', False))
        print('C', self.C, 'K', self.K, self)

    def forward(self, x):
        N, Cin, H, W = x.shape
        x = x.reshape(N, -1, self.C * self.K, H, W)
        x[:, 2, ...] = self.scales_conv(x[:, 2, ...])
        x = x.reshape(N, Cin, H, W)
        return x


class AtrousProbabilityClassifier(nn.Module):
    def __init__(self, config_ms, scale, C=3, atrous_rates_str='1,2,4'):
        raise NotImplementedError

        super(AtrousProbabilityClassifier, self).__init__()

        K = config_ms.prob.K
        Kp = non_shared_get_Kp(K, C)

        self.atrous = StackedAtrousConvs(atrous_rates_str, config_ms.Cf, Kp,
                                         kernel_size=config_ms.kernel_size,
                                         name=str(scale))
        self._repr = f'C={C}; K={K}; Kp={Kp}; rates={atrous_rates_str}'

        if global_config.get('usefinal1', False):
            print('*** Using Final')
            self.final = Final(config_ms, C, 1)
        elif global_config.get('usefinal', False):
            print('*** Using Final')
            self.final = Final(config_ms, C, 3)
        else:
            self.final = lambda x: x

        if global_config.get('initbias', False):
            K = config_ms.prob.K
            self.atrous.lin.bias = nn.Parameter(_init_bias(self.atrous.lin.bias.detach(), C, K))
            print(self.atrous.lin.bias.requires_grad)
            print('Updated bias:', self.atrous.lin.bias.reshape(-1, C, K))

            if global_config.get('usefinal1', False):
                self.final.scales_conv.bias = nn.Parameter(_init_bias(self.final.scales_conv.bias.detach(), C, K))

    def __repr__(self):
        return f'AtrousProbabilityClassifier({self._repr})'

    def forward(self, x):
        """
        :param x: NCfHW
        :return: NKpHW
        """
        y = self.atrous(x)
        y = self.final(y)
        return y


def _init_bias(bias, C, K):
    # shape of bias: Kp
    # first K are pis, second K are means, then variances, then lambdas

    bias *= 1

    # This bias is maybe not a good idea
    # probably, the PixelCNN approach of -1 1 is much smarter...
    # also, think about scale orders...

    # bias[:     C*K] += 1/K  # Pi
    # bias[C*K:  2*C*K] += (255/2 if C == 3 else 0)  # Mean
    bias[2*C*K:3*C*K] += 2  # Sigma

    # if C == 3:
    #     bias[3*C*K:      3*C*K + K] += 1/2  # Lambda g -> r
    #     bias[3*C*K + K:  3*C*K + 2*K] += 1/3  # Lambda b -> r
    #     bias[3*C*K + 2*K:3*C*K + 3*K] += 1/3  # Lambda b -> g
    #
    return bias


def test_this_shit():
    c = nn.Conv2d(64, 3*4, 3)
    x = torch.rand(1, 64, 10, 10)

    o = _init_bias(c.bias.data, 3, 1)



    print(c(x))


class ConvProbabilityClassifier(AtrousProbabilityClassifier):
    def __init__(self, config_ms, scale, C=3):
        super(ConvProbabilityClassifier, self).__init__(config_ms, scale, C, atrous_rates_str='1')

        raise NotImplementedError

        # K = config_ms.prob.K
        # self.atrous.lin.bias = nn.Parameter(_init_bias(self.atrous.lin.bias.detach(), C, K))
        # print('Updated bias:', self.atrous.lin.bias.reshape(-1, C, K))


class StackedAtrousConvs(nn.Module):
    def __init__(self, atrous_rates_str, Cin, Cout, Catrous=None,
                 bias=True, kernel_size=3, activation=None):
        super(StackedAtrousConvs, self).__init__()
        atrous_rates = self._parse_atrous_rates_str(atrous_rates_str)
        self.act = activation
        if not Catrous:
            Catrous = Cin
        self.atrous = nn.ModuleList(
                [conv(Cin, Catrous, kernel_size, rate=rate) for rate in atrous_rates])
        self.lin = conv(len(atrous_rates) * Catrous, Cout, 1, bias=bias)

        # name = name or 'stacked'
        # self.f = pe.FeatureMapSaver(f'huge_feat_normal_{name}', 'final')

        self._extra_repr = 'rates={}'.format(atrous_rates)

    @staticmethod
    def _parse_atrous_rates_str(atrous_rates_str):
        # expected to either be an int or a comma-separated string such as 1,2,4
        if isinstance(atrous_rates_str, int):
            return [atrous_rates_str]
        else:
            return list(map(int, atrous_rates_str.split(',')))

    def extra_repr(self):
        return self._extra_repr

    def forward(self, x):
        x = torch.cat([atrous(x) for atrous in self.atrous], dim=1)
        if self.act:
            x = self.act(x)
        # x = self.f(x)
        x = self.lin(x)
        return x


def test_parse():
    assert _parse_outputs_flag('means,vars,pis,coeffs') == RequiredOutputs(True, True, True, True)
    assert _parse_outputs_flag('vars,pis,coeffs') == RequiredOutputs(False, True, True, True)
    assert _parse_outputs_flag('means') == RequiredOutputs(True, False, False, False)
