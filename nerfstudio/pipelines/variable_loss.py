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
A pipeline that allows model to have variable losses with training step as an argument
"""

from dataclasses import dataclass, field
from time import time
from typing import Type
from typing_extensions import Literal

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler


@dataclass
class VariableLossPipelineConfig(VanillaPipelineConfig):
    """Variable Loss Pipeline Config"""
    _target: Type = field(default_factory=lambda: VariableLossPipeline)

class VariableLossPipeline(VanillaPipeline):
    """Pipeline that rewrites loss_dict queries to the model to also contain
    training step as an argument, thus allowing variable losses"""
    # pylint: disable=abstract-method
    config: VariableLossPipelineConfig

    def __init__(self, config: VariableLossPipelineConfig, device: str,
                 test_mode: Literal["test", "val", "inference"] = "val",
                 world_size: int = 1, local_rank: int = 0):
        super().__init__(config, device, test_mode, world_size, local_rank)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, step, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, step, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict