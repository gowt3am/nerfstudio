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
Implementation of ManhattanNeRF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import nerfacc
import torch
from nerfacc import ContractionType
from torch.nn import Parameter
from torchmetrics import (
    PeakSignalNoiseRatio,
    MeanSquaredError,
    MeanAbsoluteError,
)
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.instant_ngp_field import TCNNInstantNGPField
from nerfstudio.model_components.losses import (
    MSELossFiltered,
    OpacityLoss,
    ManhattanNormalLoss,
    angular_error_normals_degree
)

from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.data.utils.data_utils import hypersim_normals_from_ray_depths


@dataclass
class ManhattanINGPModelConfig(ModelConfig):
    """Instant NGP Model Config"""

    _target: Type = field(default_factory=lambda: ManhattanINGPModel)
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    max_num_samples_per_ray: int = 24
    """Number of samples in field evaluation."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    contraction_type: ContractionType = ContractionType.UN_BOUNDED_SPHERE
    """Contraction type used for spatial deformation of the field."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: float = 0.01
    """Minimum step size for rendering."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    background_color: Literal["random", "black", "white"] = "random"
    """The color that is given to untrained areas."""

    opacity_penalty_weight: float = 1e-3
    """Weight for opacity penalty loss."""
    min_cluster_similarity: float = 0.99
    """Minimum dot product between cluster and normal to be considered similar."""
    manhattan_orthogonal_dot_weight: float = 2e-3
    """Weight for manhattan normals to be orthogonal to each other - dot product."""
    normal_manhattan_cluster_dot_weight: float = 2e-3
    """Weight for normals to be close to manhattan cluster - dot product."""
    normal_manhattan_cluster_l1_weight: float = 2e-3
    """Weight for normals to be close to manhattan cluster - L1 loss."""
    manhattan_loss_start_step: int = 500
    """Step at which to start the manhattan losses."""
    manhattan_loss_grow_till_step: int = 2500
    """Step at which to stop growing the manhattan loss and it becomes specified value."""
    manhattan_loss_stop_step: int = -1
    """Step at which to stop using manhattan loss. -1 will never stop."""

    calc_depth_metrics: bool = False
    """Whether to calculate depth metrics."""
    calc_normal_metrics: bool = False
    """Whether to calculate normal metrics."""


class ManhattanINGPModel(Model):
    """Manhattan NeRF model - Instant NGP backbone + normal estimation and losses"""
    config: ManhattanINGPModelConfig
    field: TCNNInstantNGPField

    def __init__(self, config: ManhattanINGPModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.field = TCNNInstantNGPField(
            aabb=self.scene_box.aabb,
            contraction_type=self.config.contraction_type,
            use_appearance_embedding=self.config.use_appearance_embedding,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            contraction_type=self.config.contraction_type,
        )

        # Sampler
        vol_sampler_aabb = self.scene_box.aabb if self.config.contraction_type == ContractionType.AABB else None
        self.sampler = VolumetricSampler(
            scene_aabb=vol_sampler_aabb,
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.estimate_normals = (self.config.normal_manhattan_cluster_dot_weight > 0 or
                                self.config.normal_manhattan_cluster_l1_weight > 0 or
                                self.config.manhattan_orthogonal_dot_weight > 0)
        self.calc_normal_metrics = self.config.calc_normal_metrics and self.estimate_normals

        # losses
        self.rgb_loss = MSELossFiltered()
        self.opacity_loss = OpacityLoss()
        self.manhattan_normal_loss = ManhattanNormalLoss(
            min_cluster_similarity=self.config.min_cluster_similarity,
            manhattan_orthogonal_dot_weight=self.config.manhattan_orthogonal_dot_weight,
            normal_manhattan_dot_weight=self.config.normal_manhattan_cluster_dot_weight,
            normal_manhattan_l1_weight=self.config.normal_manhattan_cluster_l1_weight,
            start_step=self.config.manhattan_loss_start_step,
            grow_till_step=self.config.manhattan_loss_grow_till_step,
            end_step=self.config.manhattan_loss_stop_step
        )

        # metrics
        self.psnr_rgb = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_rgb = structural_similarity_index_measure
        self.lpips_rgb = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.rmse_depth = MeanSquaredError(squared=False)
        self.abs_depth = MeanAbsoluteError()
        self.angular_normal = angular_error_normals_degree

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            # TODO: needs to get access to the sampler, on how the step size is determinated at each x. See
            # https://github.com/KAIR-BAIR/nerfacc/blob/127223b11401125a9fce5ce269bb0546ee4de6e8/examples/train_ngp_nerf.py#L190-L213
            self.occupancy_grid.every_n_step(
                step=step,
                occ_eval_fn=lambda x: self.field.get_opacity(x, self.config.render_step_size),
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle, **kwargs):
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            packed_info=packed_info,
            sigmas=field_outputs[FieldHeadNames.DENSITY],
            t_starts=ray_samples.frustums.starts,
            t_ends=ray_samples.frustums.ends,
        )

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        alive_ray_mask = accumulation.squeeze(-1) > 0

        if self.estimate_normals and not kwargs.get("continuous_pixels"):
            normals, normal_valid_mask = hypersim_normals_from_ray_depths(ray_bundle, depth)
            outputs = {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
                "normals": normals,
                "normal_valid_mask": normal_valid_mask & alive_ray_mask,
                "alive_ray_mask": alive_ray_mask,  # the rays we kept from sampler
                "num_samples_per_ray": packed_info[:, 1],
            }
        else:
            outputs = {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
                "alive_ray_mask": alive_ray_mask,  # the rays we kept from sampler
                "num_samples_per_ray": packed_info[:, 1],
            }
        return outputs

    def get_metrics_dict(self, outputs, batch):
        """ Done during training, so avoid using image-level metrics such as SSIM, LPIPS, etc. """
        metrics_dict = {}
        
        metrics_dict["psnr"] = self.psnr_rgb(outputs["rgb"], batch["image"].to(self.device))
        if self.config.calc_depth_metrics:
            metrics_dict["rmse_depth"] = self.rmse_depth(outputs["depth"].squeeze(), batch["depth"].to(self.device))
            metrics_dict["abs_depth"] = self.abs_depth(outputs["depth"].squeeze(), batch["depth"].to(self.device))
        if self.calc_normal_metrics and "normals" in outputs:
            metrics_dict["angular_normal"] = self.angular_normal(outputs["normals"], batch["normals"].to(self.device))

        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, step, metrics_dict=None):
        loss_dict = {}
        mask = outputs["alive_ray_mask"]

        loss_dict["rgb"] = self.rgb_loss(batch["image"].to(self.device)[mask],
                                         outputs["rgb"][mask])
        if self.config.opacity_penalty_weight > 0:
            loss_dict["opacity"] = self.config.opacity_penalty_weight * \
                                        self.opacity_loss(outputs["accumulation"][mask])
        if self.estimate_normals and "normals" in outputs:
            loss_dict.update(self.manhattan_normal_loss(outputs["normals"][outputs["normal_valid_mask"]], step))
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        alive_ray_mask = colormaps.apply_colormap(outputs["alive_ray_mask"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        combined_alive_ray_mask = torch.cat([alive_ray_mask], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        tgt_image = torch.moveaxis(image, -1, 0)[None, ...]
        pred_rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
        
        psnr = self.psnr_rgb(tgt_image, pred_rgb)
        ssim = self.ssim_rgb(tgt_image, pred_rgb)
        lpips = self.lpips_rgb(tgt_image, pred_rgb)
        metrics_dict = {"psnr": float(psnr), "ssim": float(ssim), "lpips": float(lpips)}

        if self.config.calc_depth_metrics:
            tgt_depth = batch["depth"].to(self.device)
            pred_depth = outputs["depth"].squeeze()
            metrics_dict.update({"rmse_depth" : self.rmse_depth(tgt_depth, pred_depth),
                                 "abs_depth" : self.abs_depth(tgt_depth, pred_depth)})
        if self.calc_normal_metrics and "normals" in outputs:
            metrics_dict.update({"angular_normal" : self.angular_normal(
                        batch["normals"].to(self.device), outputs["normals"])})

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "alive_ray_mask": combined_alive_ray_mask,
        }
        return metrics_dict, images_dict
