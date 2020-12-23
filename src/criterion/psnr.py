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
import torch

import numpy as np

from helpers.quantized_tensor import SymbolTensor


_get_np = lambda x_: x_.get().detach().cpu().numpy()


def get_psnr(x: SymbolTensor, y: SymbolTensor):
    """
    Notes: Make sure that the input tensors are floats, as otherwise we get over-/underflows when we calculate MSE!
    Tested to be same as tf.image.psnr
    """
    assert x.L == y.L
    max_val = x.L - 1
    # NOTE: thats how tf.image.psnr does the mean, too: MSE over spatial, PSNR over batch
    mse = (x.get() - y.get()).pow(2).float().mean((1, 2, 3))
    assert len(mse.shape) == 1, mse.shape
    return 10. * torch.log10((max_val ** 2) / mse).mean()


def np_mse(img_a, img_b):
    """
    :param img_a: First image, as numpy array
    :param img_b: Second image, as numpy array
    :return: MSE between images
    """
    return np.mean(np.square(img_b - img_a))


def np_psnr(img_a, img_b, max_val):
    """
    :param img_a: First image, as numpy array
    :param img_b: Second image, as numpy array
    :param max_val: Maximum value that an image can have, e.g., 255. or 1., depending on normalization
    :return: PSNR between images
    """
    return 10. * np.log10((max_val ** 2) / np_mse(img_a.astype(np.float32), img_b.astype(np.float32)))



