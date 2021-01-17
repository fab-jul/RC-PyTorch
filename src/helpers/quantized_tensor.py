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
import torch
import types
from torch import Tensor


SYM_RANGE = 2


class _QTensor(object):
    def __init__(self, t: Tensor, L):
        self.t = t
        self.L = L

    def get(self) -> Tensor:
        return self.t

    def detach(self):
        self.t = self.t.detach()
        return self

    def assert_shape(self, target_shape):
        assert self.t.shape == target_shape, f'{self.t.shape} != {target_shape}'


class NormalizedTensor(_QTensor):
    def __init__(self, t: Tensor, L, centered: bool = False, sym: 'SymbolTensor' = None):
        assert t.min() >= -1 and t.max() <= 1, f'Not normalized: {t.min()}, {t.max()}'
        # assert t.dtype == torch.float32, f'Should be long: {t.type()}'
        super(NormalizedTensor, self).__init__(t, L)
        self.centered = centered
        self.sym = sym

    def to_sym(self) -> 'SymbolTensor':
        if self.sym is not None:
            self.sym.norm = self.t.detach()
            return self.sym
        t = self.t
        if not self.centered:
            t = t.add(SYM_RANGE // 2)
        t = t.mul((self.L - 1) / SYM_RANGE).round().long()
        return SymbolTensor(t, self.L, centered=self.centered)

    def __str__(self):
        return f'NormalizedTensor({self.t.shape}, L={self.L}, centered={self.centered})'

    def __repr__(self):
        return str(self)

class SymbolTensor(_QTensor):
    def __init__(self, t: Tensor, L, centered: bool = False, norm: Tensor = None):
        """
        :param t:
        :param L:
        :param centered:  If true, t is exected to be in (-L//2, L//2). when normalizing, we'll just change the range
        """
        # NOTE: 511/2 not int anymore
        assert L <= 256 or L == 511, L  # todo just to make sure math checks out
        if centered:
            self.t_range = t.min().item(), t.max().item()
            # TODO: might impact performance... could save normalize if it was created from conversion...
            assert self.t_range[0] >= (-L // 2) and self.t_range[1] <= L//2, f'Not symbol: {self.t_range}'
        else:
            assert t.min() >= 0 and t.max() < L, f'Not symbol: {t.min()}, {t.max()}'
        self.centered = centered
        self.norm_t = norm
        super(SymbolTensor, self).__init__(t, L)

    def to_norm(self) -> NormalizedTensor:
        if self.norm_t is not None:
            # TODO: we create it to prevent cycles but I'm not sure whether that's needed...
            return NormalizedTensor(self.norm_t, self.L, self.centered, sym=self)
        t = self.t.float().mul(SYM_RANGE / (self.L-1))
        if not self.centered:
            t = t.sub(SYM_RANGE // 2)
        return NormalizedTensor(t, self.L, centered=self.centered)


def test_conversion():
    import pytorch_ext as pe
    for L in (25, 256):
        s = torch.arange(L, dtype=torch.long)
        s = SymbolTensor(s, L)
        n = s.to_norm()
        so = n.to_sym()
        pe.assert_equal(s.t, so.t)

    for L in (25, 256):
        n = torch.rand(10, 10).mul(2).sub(1)
        n = NormalizedTensor(n, L)
        s = n.to_sym()
        n = s.to_norm()  # now we have a proper normalized tensor
        s = n.to_sym()
        no = s.to_norm()
        pe.assert_equal(n.t, no.t)


    for L in (511,):
        n = torch.rand(10, 10).mul(2).sub(1)
        n[0, 0] = 1
        n[0, 1] = -1
        n = NormalizedTensor(n, L, centered=True)
        s = n.to_sym()
        assert s.t.min() == -255
        assert s.t.max() == 255
        n = s.to_norm()  # now we have a proper tensor
        s = n.to_sym()
        no = s.to_norm()
        pe.assert_equal(n.t, no.t)
