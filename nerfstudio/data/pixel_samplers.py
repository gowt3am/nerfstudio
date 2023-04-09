# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code for sampling pixels.
"""

import random
import numpy as np
from einops import rearrange
from typing import Dict, Optional, Union

import torch
from torchtyping import TensorType


class PixelSampler:  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        self.kwargs = kwargs
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[TensorType] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> TensorType["batch_size", 3]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        if isinstance(mask, torch.Tensor):
            nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
            indices = nonzero_indices[chosen_indices]
        else:
            indices = torch.floor(
                torch.rand((batch_size, 3), device=device)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask" in batch:
            indices = self.sample_method(
                num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
            )
        else:
            indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        }

        assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []

        if "mask" in batch:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):

                image_height, image_width, _ = batch["image"][i].shape

                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                indices = self.sample_method(
                    num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i], device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        else:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key != "image_idx" and key != "image" and key != "mask" and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)

        assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch


class EquirectangularPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's. Assumes images are
    equirectangular and the sampling is done uniformly on the sphere.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    # overrides base method
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[TensorType] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> TensorType["batch_size", 3]:

        if isinstance(mask, torch.Tensor):
            # Note: if there is a mask, sampling reduces back to uniform sampling, which gives more
            # sampling weight to the poles of the image than the equators.
            # TODO(kevinddchen): implement the correct mask-sampling method.

            indices = super().sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:

            # We sample theta uniformly in [0, 2*pi]
            # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
            # This is done by inverse transform sampling.
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            num_images_rand = torch.rand(batch_size, device=device)
            phi_rand = torch.acos(1 - 2 * torch.rand(batch_size, device=device)) / torch.pi
            theta_rand = torch.rand(batch_size, device=device)
            indices = torch.floor(
                torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices


class PatchPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
        patch_size: side length of patch. This must be consistent in the method
        config in order for samples to be reshaped into patches correctly.
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        self.patch_size = kwargs["patch_size"]
        num_rays = (num_rays_per_batch // (self.patch_size**2)) * (self.patch_size**2)
        super().__init__(num_rays, keep_full_image, **kwargs)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overrided to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.patch_size**2)) * (self.patch_size**2)

    # overrides base method
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[TensorType] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> TensorType["batch_size", 3]:

        if mask:
            # Note: if there is a mask, sampling reduces back to uniform sampling
            indices = super().sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            sub_bs = batch_size // (self.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.patch_size, image_width - self.patch_size],
                device=device,
            )

            indices = indices.view(sub_bs, 1, 1, 3).broadcast_to(sub_bs, self.patch_size, self.patch_size, 3).clone()

            yys, xxs = torch.meshgrid(
                torch.arange(self.patch_size, device=device), torch.arange(self.patch_size, device=device)
            )
            indices[:, ..., 1] += yys
            indices[:, ..., 2] += xxs

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        return indices

class TrianglePixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples upper left triangular
    patches uniform randomly from all images. Useful for normal estimation later
    TODO: Single image pixel sampling not implemented here
        
        With dilation rate 0:
        ____|_x2_|____
        _x3_|_x1_|____
            |    |  
        
        With dilation rate 1:
        ____|____|_x2_|____|____
        ____|____|____|____|____
        _x3_|____|_x1_|____|____
        ____|____|____|____|____  
        ____|____|____|____|____
    Args:
        num_rays_per_batch: number of rays to sample per batch (will be divided by 3 for triangles)
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """
    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs) -> None:
        # Make number of rays per batch is divisible by 3
        num_rays_per_batch = (num_rays_per_batch // 3) * 3
        super().__init__(num_rays_per_batch, keep_full_image, **kwargs)
        self.dilaion_rate = kwargs["dilation_rate"]
        self.height = kwargs["height"]
        self.width = kwargs["width"]

        self.setup_valid_x123_idxs()

    def setup_valid_x123_idxs(self):
        self.num_indices = self.height * self.width
        img_indices = np.arange(0, self.num_indices, 1, dtype=np.int64)
        img_indices = rearrange(img_indices, '(h w) -> h w', h = self.height)
        # ____|_x2_|____
        # _x3_|_x1_|____
        #     |    |  
        # Skip boundaries and select only valid values
        self.valid_idx = {
            'x1': rearrange(img_indices[1:-1, 1:-1], 'h w -> (h w)'),
            'x2': rearrange(img_indices[:-2, 1:-1], 'h w -> (h w)'),
            'x3': rearrange(img_indices[1:-1, :-2], 'h w -> (h w)')}
    
    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overrided to make it divisible by 3.
        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // 3) * 3

    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[TensorType] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> TensorType["batch_size", 3]:
        if mask is not None:
            raise NotImplementedError("Masking not implemented for TrianglePixelSampler")
        
        image_indices = np.random.choice(num_images, batch_size//3)
        triangle_indices = np.random.choice(self.valid_idx['x1'].shape[0], batch_size//3)
        x1_indices = self.valid_idx['x1'][triangle_indices]
        x2_indices = self.valid_idx['x2'][triangle_indices]
        x3_indices = self.valid_idx['x3'][triangle_indices]
            
        image_indices = np.concatenate((image_indices,)*3, axis=0)
        if self.dilaion_rate > 0:
            # Dilate all pixels to make the triangle bigger
            x1_indices_new = (x1_indices + self.dilaion_rate*self.width)
            x2_indices_new = (x2_indices - self.dilaion_rate*self.width)
            x3_indices_new = x3_indices - self.dilaion_rate

            x1_indices = np.where(x1_indices_new < self.num_indices, x1_indices_new, x1_indices)
            x2_indices = np.where(x2_indices_new >= 0, x2_indices_new, x2_indices)    
            x3_indices = np.where((x3_indices_new // self.width) == (x3_indices // self.width),
                                  x3_indices_new, x3_indices)
        pixel_indices = np.concatenate((x1_indices, x2_indices, x3_indices), axis=0)
        
        return torch.from_numpy(np.stack((image_indices, pixel_indices // self.width,
                                pixel_indices % self.width), axis=1)).to(device)