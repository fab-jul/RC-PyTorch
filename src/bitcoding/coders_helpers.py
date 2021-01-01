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

Very thin wrapper around DiscretizedMixLogisticLoss.cdf_step_non_shared that keeps track of c_cur

"""

import torch

from criterion.logistic_mixture import DiscretizedMixLogisticLoss, CDFOut
from modules.prob_clf import NetworkOutput


class CodingCDFNonshared(object):
    def __init__(self, network_out: NetworkOutput, dmll: DiscretizedMixLogisticLoss, x_range, centered_x: bool):
        """
        :param network_out: predicted distribution,
        :param dmll:
        """
        assert centered_x  # not implemented for x>=0
        self.network_out = network_out
        self.dmll = dmll
        self.c_cur = 0
        a, b = x_range
        self.targets = self.dmll.targets[dmll.L // 2 + a:dmll.L//2 + b + 2]
        assert len(self.targets) == b - a + 2, (len(self.targets), a, b)  # Lp

    def get_next_C(self, decoded_x) -> CDFOut:
        """
        Get CDF to encode/decode next channel
        :param decoded_x: NCHW
        :return: C_cond_cur, NHWL'
        """
        C_Cur = self.dmll.cdf_step_non_shared(self.network_out, self.c_cur, self.targets, decoded_x)
        self.c_cur += 1
        return C_Cur

