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
import vis
import torch
import pytorch_ext as pe
from torch import nn

from helpers import cin_bins
from helpers.global_config import global_config
from helpers.quantized_tensor import NormalizedTensor
from modules import edsr, act, prob_clf, ups
from modules.conditional_instance_norm import ConditionalInstanceNorm2d
from modules.gdn import GDN


class SequentialWithSkip(nn.Module):
    def __init__(self, body, final):
        super(SequentialWithSkip, self).__init__()
        self.body = body
        self.final = final

    def forward(self, x):
        x = self.body(x) + x
        return self.final(x)


class SideInformationNetwork(nn.Module):
    def __init__(self, Ccond):
        super(SideInformationNetwork, self).__init__()

        self.net = nn.Sequential(
                nn.Conv2d(3, Ccond, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(Ccond, Ccond, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(Ccond, Ccond, kernel_size=3),
                nn.ReLU(inplace=True))

    def forward(self, x_residual):
        return self.net(x_residual).mean((2, 3))  # mean over spatial

def test_side():
    B, C = 5, 128
    x = torch.rand(B, 3, 16, 16)
    n = SideInformationNetwork(C)
    x_out = n(x)
    assert x_out.shape == (B, C), x_out.shape
    print(x_out)
    cc = ConditionalConvolution(5, 5, C, 3)

    x2 = torch.rand(B, 5, 16, 16)
    x3 = cc(x2, x_out)


class ConditionalConvolution(nn.Module):
    def __init__(self, Cin, Cout, Ccond, kernel_size, activation=None):
        super(ConditionalConvolution, self).__init__()
        self.conv = pe.default_conv(Cin, Cout, kernel_size, bias=False)
        self.to_scale = nn.Sequential(
                nn.Linear(Ccond, Cout),
                nn.ReLU(inplace=True))
        self.to_bias = nn.Sequential(
                nn.Linear(Ccond, Cout),
                nn.ReLU(inplace=True))
        self.activation = activation

    def forward(self, x, cond_in):
        """
        :param x: BCcHW
        :param cond_in: BCc
        :return:
        """
        B, Ccond, _, _ = x.shape
        scale = 1 - self.to_scale(cond_in).view(B, Ccond, 1, 1)
        bias = self.to_bias(cond_in).view(B, Ccond, 1, 1)
        x = scale * self.conv(x) + bias
        if self.activation is not None:
            x = self.activation(x)
        return x



class EnhancementNetwork(vis.summarizable_module.SummarizableModule):
    def __init__(self, config_en):
        super(EnhancementNetwork, self).__init__()
        self.config_en = config_en

        Cf = config_en.Cf
        kernel_size = config_en.kernel_size
        n_resblock = config_en.n_resblock

        more_gdn = global_config.get('more_gdn', False)
        more_act = global_config.get('more_act', False)

        act_body = act.make(Cf, inverse=False)
        act_tail = act.make(Cf, inverse=False)

        self.head = pe.default_conv(3, Cf, 3)
        if more_act:
            self.head = nn.Sequential(self.head, act_body)

        self._down_up = global_config.get('down_up', None)
        if self._down_up:
            self.down = pe.default_conv(Cf, Cf, global_config.get('fw_du', 3), stride=2)
            if more_gdn:
                self.down = nn.Sequential(self.down, GDN(Cf))
            if more_act:
                self.down = nn.Sequential(self.down, act_body)
        else:
            self.down = lambda x: x

        self.unet_skip_conv = None
        if global_config.get('unet_skip', None):
            self.unet_skip_conv = nn.Sequential(
                    pe.default_conv(2*Cf, Cf, 3),
                    nn.ReLU(inplace=True))

        assert not global_config.get('learned_skip', False)

        # Cf_resnet = global_config.get('Cf_resnet', Cf)
        # print('*** Cf_resnet ==', Cf_resnet5)

        norm_cls = None
        if global_config.get('inorm', False):
            print('***Using Instance Norm!')
            norm_cls = lambda: nn.InstanceNorm2d(Cf, affine=True)

        if global_config.get('gdn', False):
            norm_cls = lambda: GDN(Cf)

        use_norm_for_long = not global_config.get('no_norm_final', False)
        if not use_norm_for_long:
            print('*** no norm for final')

        def make_res_block(_act, _use_norm=True):
            return edsr.ResBlock(
                pe.default_conv, Cf, kernel_size, act=_act,
                norm_cls=norm_cls if _use_norm else None,
                res_scale=global_config.get('res_scale', 0.1))

        norm_in_body = True

        if global_config.get('gdn_as_nl', False):
            print('*** GDN as non linearity!')
            norm_cls = None
            act_body = GDN(Cf)
            if not global_config.get('gdnfreetail', False):
                act_tail = GDN(Cf)
            norm_in_body = False
            use_norm_for_long = False

        m_body = [
            make_res_block(act_body, norm_in_body)
            for _ in range(n_resblock)
        ]
        m_body.append(pe.default_conv(Cf, Cf, kernel_size))
        self.body = nn.Sequential(*m_body)

        if self._down_up:
            if self._down_up == 'deconv':
                up = ups.DeconvUp(config_en)
            elif self._down_up == 'nn':
                up = ups.ResizeConvUp(config_en)
            else:
                up = edsr.Upsampler(pe.default_conv, 2, Cf, act=False)

            if more_gdn:
                up = nn.Sequential(up, GDN(Cf, inverse=True))
            if more_act:
                up = nn.Sequential(up, act_body)

            print('*** DownUp, adding', up)
            self.after_skip = up
        else:
            self.after_skip = lambda x: x

        tail_networks = {}
        if global_config.get('deeptails', False):
            raise NotImplemented
            # num_blocks = global_config['deeptails']
            # for name in ('sigmas', 'means'):
            #     tail_networks[name] = lambda: SequentialWithSkip(
            #             body=nn.Sequential(*[make_res_block(nn.LeakyReLU(inplace=True))
            #                                  for _ in range(num_blocks)]),
            #             final=pe.default_conv(Cf, prob_clf.ProbClfTail.get_cout(config_en), 1))

        def _tail(fw_=3):
            if global_config.get('atrous', None):
                print('Atrous Tail')
                assert 'long_sigma' in global_config
                assert 'long_means' in global_config
                return [
                    prob_clf.StackedAtrousConvs(
                            atrous_rates_str='1,2,4',
                            Cin=Cf, Cout=prob_clf.ProbClfTail.get_cout(config_en), Catrous=Cf//2,
                            bias=False, activation=nn.LeakyReLU(inplace=True))]
            else:  # default so far
                return [
                    pe.default_conv(Cf, Cf, fw_),
                    nn.LeakyReLU(inplace=True),
                    pe.default_conv(Cf, prob_clf.ProbClfTail.get_cout(config_en), 1),  # final 1x1
                ]

        if global_config.get('long_sigma', False):
            fw_sigma = global_config.get('fw_s', 5)
            print('filter_width for sigma =', fw_sigma)
            modules = [make_res_block(act_tail, use_norm_for_long),
                       *_tail(fw_sigma)]
            if global_config.get('fc2', False):
                print('Adding another 1x1 conv!')
                modules.insert(-1, pe.default_conv(Cf, Cf, 1))
            tail_networks['sigmas'] = lambda: pe.FeatureMapSaverSequential(
                    *modules,
                    saver=None  # pe.FeatureMapSaver()
            )
            print('Did set tail_networks.sigmas')
        if global_config.get('long_means', False):
            tail_networks['means'] = lambda: pe.FeatureMapSaverSequential(
                    make_res_block(act_tail, use_norm_for_long),
                    *_tail()
                    # saver=self.savers['final_sigmas'], idx=-2
            )
            print('Did set tail_networks.means')
        if global_config.get('long_pis', False):
            tail_networks['pis'] = lambda: pe.FeatureMapSaverSequential(
                    make_res_block(act_tail, use_norm_for_long),
                    pe.default_conv(Cf, Cf, 3),  # no crazy smoothing
                    nn.LeakyReLU(inplace=True),  # a non linearity
                    pe.default_conv(Cf, prob_clf.ProbClfTail.get_cout(config_en), 1),  # final 1x1
                    saver=None
            )
            print('Did set tail_networks.pis')
        if global_config.get('long_lambdas', False):
            tail_networks['lambdas'] = lambda: nn.Sequential(
                    make_res_block(act_tail, use_norm_for_long),
                    pe.default_conv(Cf, Cf, 3),  # no crazy smoothing
                    nn.LeakyReLU(inplace=True),  # a non linearity
                    pe.default_conv(Cf, prob_clf.ProbClfTail.get_cout(config_en), 1),  # final 1x1
            )
            print('Did set tail_networks.lambdas')

        if global_config.get('longer_lambda', False):
            tail_networks['lambdas'] = lambda: nn.Sequential(
                    pe.default_conv(Cf, Cf, 3),
                    nn.LeakyReLU(inplace=True),
                    pe.default_conv(Cf, prob_clf.ProbClfTail.get_cout(config_en), 1),  # final 1x1
            )
            print('Did set tail_networks.lambdas')

        self.side_information_mode = False
        if global_config.get('side_information', False):
            print('*** Using side_information!')
            self.side_information_mode = True
            Ccond = global_config['side_information']
            self.side_information_net = SideInformationNetwork(Ccond)
            self.side_information_conv = ConditionalConvolution(Cf, Cf, Ccond, kernel_size,
                                                                activation=nn.LeakyReLU(inplace=True))

        print('Setting tail_networks[', tail_networks.keys(), ']')
        self.tail = prob_clf.ProbClfTail.from_config(config_en, tail_networks=tail_networks)

    def forward(self, x_n: NormalizedTensor, side_information=None) -> prob_clf.NetworkOutput:
        self.summarizer.register_images('train', {'input': lambda: x_n.to_sym().get().to(torch.uint8)}, only_once=True)
        x = x_n.get()
        x = self.head(x)
        x_after_head = x
        x = self.down(x)
        x = self.body(x) + x
        x = self.after_skip(x)  # goes up again

        if self.unet_skip_conv is not None:
            x = self.unet_skip_conv(torch.cat((x, x_after_head), dim=1))

        if self.side_information_mode:
            x = self.side_information_conv(x, side_information)

        return self.tail(x)

    def extract_side_information(self, x_r: NormalizedTensor, x_l: NormalizedTensor):
        """ :return (B, Ccond) """
        if not self.side_information_mode:
            return None
        x_residual = x_r.get() - x_l.get()
        return self.side_information_net(x_residual)

