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
from torch import nn

from helpers.quantized_tensor import NormalizedTensor
from vis.summarizable_module import SummarizableModule


class MSE(SummarizableModule):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x: NormalizedTensor, predicted_means):
        x.assert_shape(predicted_means.shape)
        x = x.get()
        return self.mse(predicted_means, x)

