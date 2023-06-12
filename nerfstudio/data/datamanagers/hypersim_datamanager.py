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
HyperSim datamanager.
"""

from dataclasses import dataclass, field
from typing import Type, List, Any, Union, Literal

import torch
from torchtyping import TensorType
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datasets.hypersim_dataset import HyperSimDataset
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PixelSampler, TrianglePixelSampler

@dataclass
class HyperSimDataManagerConfig(VanillaDataManagerConfig):
    """A hypersim datamanager - required to use with .setup()"""
    _target: Type = field(default_factory=lambda: HyperSimDataManager)
    
    labels: List[str] = field(default_factory=lambda: ["normals", "depth"])
    """Labels/Files to load"""
    ray_sampling_strategy: str = "triangle"
    """The ray sampling strategy to use. Options are "triangle" and "uniform"."""
    dilation_rate: int = 2
    """The dilation factor to use for the triangle pixel sampler."""


class HyperSimDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Data manager implementation for hypersim dataset.
    Args:
        config: the DataManagerConfig used to instantiate class
    """
    config: HyperSimDataManagerConfig

    def __init__(self,
        config: HyperSimDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)
        self.H_orig = self.train_dataparser_outputs.metadata["H_orig"]
        self.W_orig = self.train_dataparser_outputs.metadata["W_orig"]
        self.H = int(self.H_orig * self.config.camera_res_scale_factor)
        self.W = int(self.W_orig * self.config.camera_res_scale_factor)

    def _get_pixel_sampler(  # pylint: disable=no-self-use
        self, dataset: InputDataset, *args: Any, **kwargs: Any) -> PixelSampler:
        """Infer which pixel sampler to use."""
        if self.config.ray_sampling_strategy == "triangle":
            if self.test_tuning or self.pregen_random_views or self.on_the_fly_random_views:
                raise ValueError("Triangle pixel sampler is not supported for test tuning with masks currently, use uniform.")
            return TrianglePixelSampler(*args, **kwargs, height = dataset.H,
                    width = dataset.W, dilation_rate = self.config.dilation_rate)
        elif self.config.ray_sampling_strategy == "uniform":
            return PixelSampler(*args, **kwargs)
        else:
            raise ValueError(f"Invalid ray sampling strategy: {self.config.ray_sampling_strategy}")

    def create_train_dataset(self) -> HyperSimDataset:
        return HyperSimDataset(dataparser_outputs=self.train_dataparser_outputs,
                               scale_factor=self.config.camera_res_scale_factor,
                               labels=self.config.labels, test_tuning=self.test_tuning,
                               pregen_random_views=self.pregen_random_views,
                               on_the_fly_random_views=self.on_the_fly_random_views)

    def create_eval_dataset(self) -> HyperSimDataset:
        return HyperSimDataset(dataparser_outputs=self.dataparser.get_dataparser_outputs(
            split=self.test_split), scale_factor=self.config.camera_res_scale_factor,
            labels=self.config.labels, test_tuning=False, pregen_random_views=False,
            on_the_fly_random_views=False)

    def generate_random_views(self, num_views: int) -> TensorType:
        """Generate random views for training.
        Args:
            num_views: number of random views to generate
        """
        return self.train_dataset.generate_random_views(num_views)