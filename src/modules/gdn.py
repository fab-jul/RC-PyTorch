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

Implement balle's GDN layer in PyTorch.
"""
import types

import torch
from torch import nn



class _LowerBoundFunction(torch.autograd.Function):
    # Implements LowerBound with identity_if_towards
    # https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/ops/math_ops.py#L132
    #
    # From there:
    #
    # Same as `tf.maximum`, but with helpful gradient for `inputs < bound`.
    #   This function behaves just like `tf.maximum`, but the behavior of the gradient
    #   with respect to `inputs` for input values that [are lower than bound are different]
    #   The gradient is replaced with the identity
    #   function, but only if applying gradient descent would push the values of
    #   `inputs` towards the bound. For gradient values that push away from the bound,
    #   the returned gradient is still zero.
    #
    @staticmethod
    def forward(ctx, input, bound):
        bound = torch.ones_like(input) * bound
        ctx.save_for_backward(input, bound)
        return torch.max(input, bound)

    @staticmethod
    def backward(ctx, grad_output):
        input, bound = ctx.saved_tensors

        # <0 because p = p - \lambda * p.grad
        pass_through = (input >= bound) | (grad_output < 0)

        return pass_through.type(grad_output.dtype) * grad_output, None


def NonNegativeParameter(initial_value,
                         minimum=0., reparam_offset=2 ** -18):
    """
    Return a nn.Parameter with a function `constrained` that returns a non-negative tensor.
    """
    pedestal = torch.tensor(reparam_offset ** 2, requires_grad=False)
    bound = (minimum + reparam_offset ** 2) ** .5
    reparam_initial_value = torch.sqrt(torch.max(initial_value + pedestal, pedestal))

    p = nn.Parameter(reparam_initial_value.data, requires_grad=True)

    def _reparam(self):
        # self.data = self.clamp(min=bound).pow(2) - pedestal
        lower_bound = _LowerBoundFunction.apply
        return lower_bound(self, bound).pow(2) - pedestal

    # Add method `constrained` to this parameter, that reparametrizes it
    p.constrained = types.MethodType(_reparam, p)
    p.constrained()
    return p



class GDN(nn.Module):
    def __init__(self,
                 C,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1):
        super(GDN, self).__init__()
        self.inverse = inverse

        self.C = C

        self.beta, self.gamma = self._create_params(beta_min, gamma_init)

    def _create_params(self, beta_min, gamma_init):
        beta = NonNegativeParameter(
                initial_value=torch.ones(self.C, dtype=torch.float32),
                minimum=beta_min)
        gamma = NonNegativeParameter(
                initial_value=torch.eye(self.C, dtype=torch.float32) * gamma_init,
                minimum=0)
        return beta, gamma

    def forward(self, x):
        C = self.C
        # gamma is (C, C)
        gamma = self.gamma.constrained().view(C, C, 1, 1)
        # beta is (C,)
        beta = self.beta.constrained()

        return self._forward(x, gamma, beta)

    def _forward(self, x, gamma, beta):
        # VALID is default for PyTorch
        norm_pool = nn.functional.conv2d(x ** 2, gamma, bias=beta)
        if self.inverse:
            norm_pool = torch.sqrt(norm_pool)
        else:
            norm_pool = torch.rsqrt(norm_pool)
        return x * norm_pool

    def __repr__(self):
        return f'{"I" if self.inverse else ""}GDN(C={self.C})'


