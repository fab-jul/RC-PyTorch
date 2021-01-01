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
from torchvision import transforms

from dataloaders import images_loader
from helpers.global_config import global_config


def get_test_dataset_transform(crop):
    img_to_tensor_t = [
        images_loader.IndexImagesDataset.to_tensor_uint8_transform()]
    if global_config.get('ycbcr', False):
        print('Adding ->YCbCr to Testset')
        t = transforms.Lambda(lambda pil_img: pil_img.convert('YCbCr'))
        img_to_tensor_t.insert(0, t)
    if crop:
        print(f'Cropping Testset: {crop}')
        img_to_tensor_t.insert(0, transforms.CenterCrop(crop))
    return transforms.Compose(img_to_tensor_t)

