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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import random
import torch
import torch.nn.functional as F
from torchtyping import TensorType
from torch.nn import Parameter
from torchmetrics import (
    PeakSignalNoiseRatio,
    MeanSquaredError,
    MeanAbsoluteError,
)
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal
from einops import rearrange

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.losses import (
    MSELossFiltered,
    OpacityLoss,
    ManhattanNormalLoss,
    NormalSupervisionLoss,
    angular_error_normals_degree,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.data.utils.data_utils import hypersim_normals_from_ray_depths


@dataclass
class ManhattanNerfactoModelConfig(ModelConfig):
    """Manhattan Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: ManhattanNerfactoModel)
    near_plane: float = 0.0
    """How far along the ray to start sampling."""
    far_plane: float = 20.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not. If disabled, also uses AABBCollider instead of NearFar"""
    use_appearance_embedding: bool = False
    """Whether to use appearance embedding."""
    
    estimate_normal_from_depth: bool = True
    """Whether to estimate normal from depth, or render them as derivative of density (RefNeRF)"""
    normal_gt_supervision_loss: bool = False
    """Whether to use normal GT supervision loss (L1 + dot) or not."""
    opacity_penalty_weight: float = -1.0 #1e-3
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

    calc_depth_metrics: bool = True
    """Whether to calculate depth metrics."""
    calc_normal_metrics: bool = True
    """Whether to calculate normal metrics."""
    use_affine_illumination_modeling: bool = False
    """Whether to use affine illumination modeling or not for random views"""
    rendered_depth_new_view_start_step: int = 1000
    """Step at which to start using rendered depth to create random views and calculate loss"""
    use_only_manhattan_depth_for_rendered_views: bool = False
    """Whether to use only manhattan-normal pixels to create random views or not"""
    use_depth_consistency_loss: bool = False
    """Whether to use depth consistency loss for rendered_views or not"""


class ManhattanNerfactoModel(Model):
    """Manhattan Nerfacto model
    In addition to Manhattan Priors, contains some differences from base Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: ManhattanNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.test_tuning = self.kwargs["test_tuning"]
        self.pregen_random_views = self.kwargs["pregen_random_views"]
        self.on_the_fly_random_views = self.kwargs["on_the_fly_random_views"]
        self.rendered_depth_new_views = self.kwargs["rendered_depth_new_views"]
        self.num_test_data = self.kwargs["num_test_data"]
        self.num_random_views = self.kwargs["num_random_views"]
        self.H_orig = self.metadata["H_orig"]
        self.W_orig = self.metadata["W_orig"]
        self.M_cam_from_uv = self.metadata["M_cam_from_uv"]
        self.M_ndc_from_cam = self.metadata["M_ndc_from_cam"].cuda()
        self.M_uv_from_ndc = self.metadata["M_uv_from_ndc"].cuda()
        self.distance_per_z = self.metadata["distance_per_z"].cuda()
        if self.rendered_depth_new_views:
            self.prepare_random_view_generation()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=32 if self.config.use_appearance_embedding else 0
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        if self.config.disable_scene_contraction:
            self.collider = AABBBoxCollider(self.scene_box)
            # self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        else:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

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
        self.normal_supervision_loss = NormalSupervisionLoss(
            dot_weight=1e-3, l1_weight=1e-3, start_step=1500, grow_till_step=3500, end_step=-1)
        self.use_manhattan_losses = (self.config.normal_manhattan_cluster_dot_weight > 0 or
                                     self.config.normal_manhattan_cluster_l1_weight > 0 or
                                     self.config.manhattan_orthogonal_dot_weight > 0)
        self.use_normal_gt_losses = self.config.normal_gt_supervision_loss

        # metrics
        self.psnr_rgb = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_rgb = structural_similarity_index_measure
        self.lpips_rgb = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.rmse_depth = MeanSquaredError(squared=False)
        self.abs_depth = MeanAbsoluteError()
        self.angular_normal = angular_error_normals_degree

        # Parameters for gain compensation for test set tuning
        if self.config.use_affine_illumination_modeling:
            if self.test_tuning:
                self.alpha = Parameter(torch.ones(self.num_test_data))
                self.beta = Parameter(torch.zeros(self.num_test_data))
            elif self.pregen_random_views or self.on_the_fly_random_views:
                self.alpha = Parameter(torch.ones(self.num_random_views))
                self.beta = Parameter(torch.zeros(self.num_random_views))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        if self.config.use_affine_illumination_modeling:
            param_groups["fields"] = list(self.field.parameters()) + [self.alpha, self.beta]
        else:
            param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle, batch, **kwargs):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=True)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.estimate_normal_from_depth and not kwargs.get("continuous_pixels"):
            # Estimating normals from depth (need ray_bundle to be ordered as P1 | P2 | P3)
            normals, normal_valid_mask = hypersim_normals_from_ray_depths(ray_bundle, depth)
            outputs["normals"] = normals
            outputs["normal_valid_mask"] = normal_valid_mask
        else: 
            # Rendering normals as derivative of field density (by default when evaluation as continuous pixels)
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )
            outputs["normals"] = normals
            outputs["normal_valid_mask"] = torch.ones_like(normals[..., 0]) > 0
            outputs["normals_vis"] = self.normals_shader(normals)

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        if self.rendered_depth_new_views and self.training and batch is not None and \
                kwargs.get("step", -1) >= self.config.rendered_depth_new_view_start_step:
            rand_pose = self.sparf_pose_interpolation(batch["pose"], batch["closest_pose"])
            dst_indices, mask, z_depth = self.warp_rgbd_to_new_pose(batch["indices"], outputs["depth"],
                                                                      batch["pose"], rand_pose)

            if self.config.use_only_manhattan_depth_for_rendered_views and "normals" in outputs:
                normal_clusters = self.manhattan_normal_loss.get_top_3_cluster_associations(outputs["normals"])
                dst_indices = dst_indices[normal_clusters != 0].contiguous()
                mask = mask[normal_clusters != 0].contiguous()
                z_depth = z_depth[normal_clusters != 0].contiguous()
                rgb_tgt = batch["image"].to(self.device)[normal_clusters != 0][mask].contiguous()
            else:
                rgb_tgt = batch["image"].to(self.device)[mask].contiguous()
            
            # Generate rays for rand_pose camera, forward propagate through field and render its RGB
            dst_indices = dst_indices[mask].contiguous().long()
            z_depth = z_depth[mask].contiguous()
            dst_ray_bundle = self.ray_generator(dst_indices, rand_pose)

            # Forward propagate through field and render its RGB
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(dst_ray_bundle, density_fns=self.density_fns)
            field_outputs = self.field(ray_samples, compute_normals=False)
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            weights_list.append(weights)
            ray_samples_list.append(ray_samples)
            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            
            if self.config.use_depth_consistency_loss:
                depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
                dst_distance = z_depth.squeeze() / self.distance_per_z[dst_indices[:, 1], dst_indices[:, 2]]
                outputs["depth_consistency_loss"] = self.rgb_loss(depth.squeeze(), dst_distance)
            # accumulation = self.renderer_accumulation(weights=weights)
            outputs["rand_rgb_loss"] = self.rgb_loss(rgb_tgt, rgb)
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        if self.training:
            if self.test_tuning or self.pregen_random_views:
                if self.test_tuning:
                    tgt = batch["pregenerated"].to(self.device)
                else:
                    tgt = batch["image"].to(self.device)
                if self.config.use_affine_illumination_modeling:
                    indices = batch["indices"].to(self.device)
                    a = self.alpha[indices[:, 0]].unsqueeze(1)
                    b = self.beta[indices[:, 0]].unsqueeze(1)
                    tgt = a * tgt + b
            elif self.on_the_fly_random_views:
                if self.config.use_affine_illumination_modeling:
                    indices = [self.rand_indices_dict[x] for x in batch["indices"][:, 0].cpu().numpy()]
                    a = [self.alpha[x] if x != -1 else 1.0 for x in indices]
                    b = [self.beta[x] if x != -1 else 0.0 for x in indices]
                    a = torch.tensor(a).float().unsqueeze(1).to(self.device)
                    b = torch.tensor(b).float().unsqueeze(1).to(self.device)
                    tgt = a * batch["image"].to(self.device) + b
                else:
                    tgt = batch["image"].to(self.device)
            else:
                tgt = batch["image"].to(self.device)
        else:
            tgt = batch["image"].to(self.device)

        metrics_dict["psnr"] = self.psnr_rgb(outputs["rgb"], tgt)
        if self.config.calc_depth_metrics and "depth" in outputs and "depth" in batch:
            metrics_dict["rmse_depth"] = self.rmse_depth(outputs["depth"].squeeze(), batch["depth"].to(self.device))
            metrics_dict["abs_depth"] = self.abs_depth(outputs["depth"].squeeze(), batch["depth"].to(self.device))
        if self.config.calc_normal_metrics and "normals" in outputs and "normals" in batch:
            metrics_dict["angular_normal"] = self.angular_normal(outputs["normals"], batch["normals"].to(self.device))
            # metrics_dict.update(self.normal_supervision_loss(outputs["normals"][outputs["normal_valid_mask"]],
            #                                                  batch["normals"][outputs["normal_valid_mask"]].to(self.device), 4250))
            # metrics_dict.update(self.manhattan_normal_loss(outputs["normals"][outputs["normal_valid_mask"]], 4250))

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, step, metrics_dict=None):
        loss_dict = {}
        if self.training:
            if self.test_tuning or self.pregen_random_views:
                if self.test_tuning:
                    tgt = batch["pregenerated"].to(self.device)
                else:
                    tgt = batch["image"].to(self.device)
                if self.config.use_affine_illumination_modeling:
                    indices = batch["indices"].to(self.device)
                    a = self.alpha[indices[:, 0]].unsqueeze(1)
                    b = self.beta[indices[:, 0]].unsqueeze(1)
                    tgt = a * tgt + b
                    # Adding a normalizing loss term to keep alpha and beta close to 1, 0
                    loss_dict["alpha_beta"] = 0.01 * (torch.mean(torch.abs(a - 1.0)) + torch.mean(torch.abs(b)))
            elif self.on_the_fly_random_views:
                if self.config.use_affine_illumination_modeling:
                    indices = [self.rand_indices_dict[x] for x in batch["indices"][:, 0].cpu().numpy()]
                    a = [self.alpha[x] if x != -1 else 1.0 for x in indices]
                    b = [self.beta[x] if x != -1 else 0.0 for x in indices]
                    a = torch.tensor(a).float().unsqueeze(1).to(self.device)
                    b = torch.tensor(b).float().unsqueeze(1).to(self.device)
                    tgt = a * batch["image"].to(self.device) + b
                    # Adding a normalizing loss term to keep alpha and beta close to 1, 0
                    loss_dict["alpha_beta"] = 0.01 * (torch.mean(torch.abs(a - 1.0)) + torch.mean(torch.abs(b)))
                else:
                    tgt = batch["image"].to(self.device)
            else:
                tgt = batch["image"].to(self.device)
        else:
            tgt = batch["image"].to(self.device)

        loss_dict["rgb"] = self.rgb_loss(tgt, outputs["rgb"])
        if self.config.opacity_penalty_weight > 0:
            loss_dict["opacity"] = self.config.opacity_penalty_weight * self.opacity_loss(outputs["accumulation"])
        
        if self.use_manhattan_losses and "normals" in outputs and "normals" in batch:
            loss_dict.update(self.manhattan_normal_loss(outputs["normals"][outputs["normal_valid_mask"]], step))
        
        if self.use_normal_gt_losses and "normals" in outputs and "normals" in batch:
            loss_dict.update(self.normal_supervision_loss(outputs["normals"][outputs["normal_valid_mask"]],
                                                          batch["normals"][outputs["normal_valid_mask"]].to(self.device), step))
        
        # if "rendered_orientation_loss" in outputs:
        #     loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(outputs["rendered_orientation_loss"])
        
        if self.training:
            if "rand_rgb_loss" in outputs:
                loss_dict["rand_rgb_loss"] = outputs["rand_rgb_loss"]
            if "depth_consistency_loss" in outputs:
                loss_dict["depth_consistency_loss"] = outputs["depth_consistency_loss"]
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
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

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr_rgb(image, rgb)
        ssim = self.ssim_rgb(image, rgb)
        lpips = self.lpips_rgb(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr), "ssim": float(ssim), "lpips": float(lpips)}

        if self.config.calc_depth_metrics and "depth" in outputs and "depth" in batch:
            tgt_depth = batch["depth"].to(self.device)
            pred_depth = outputs["depth"].squeeze()
            metrics_dict.update({"rmse_depth" : self.rmse_depth(tgt_depth, pred_depth),
                                 "abs_depth" : self.abs_depth(tgt_depth, pred_depth)})
        
        if self.config.calc_normal_metrics and "normals" in outputs and "normals" in batch:
            tgt_normals = batch["normals"].to(self.device)
            pred_normals = outputs["normals"]
            metrics_dict.update({"angular_normal" : self.angular_normal(tgt_normals, pred_normals)})
            # metrics_dict.update(self.normal_supervision_loss(pred_normals, tgt_normals, 4250))

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        if "normals_vis" in outputs:
            if "normals" in batch:
                normal_gt = (batch["normals"].to(self.device) + 1.0) / 2.0
                combined_normal = torch.cat([normal_gt, outputs["normals_vis"]], dim=1)
            else:
                combined_normal = outputs["normals_vis"]
            images_dict["normals"] = combined_normal
        return metrics_dict, images_dict

    def reset_illumination_parameters(self, rand_indices_dict) -> None:
        self.rand_indices_dict = rand_indices_dict
        if self.config.use_affine_illumination_modeling:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if 'alpha' in name:
                        param.copy_(torch.ones(self.num_random_views))
                    elif 'beta' in name:
                        param.copy_(torch.zeros(self.num_random_views))
    
    #### Below are functions for rendering new images using rendered depth and backpropagating
    def prepare_random_view_generation(self):
        u = np.linspace(-1.0 + 1.0/self.W_orig, 1.0 - 1.0/self.W_orig, self.W_orig)
        v = np.linspace(-1.0 + 1.0/self.H_orig, 1.0 - 1.0/self.H_orig, self.H_orig)
        # Reverse vertical coordinate because [H=0, W=0] corresponds to (u=-1, v=1)
        u, v = np.meshgrid(u, v[::-1])
        uv = torch.as_tensor(np.dstack((u, v, np.ones_like(u))), dtype=torch.float32)    # (H, W, 3)
        uv = rearrange(uv, 'h w c -> (h w) c')

        xyz = (self.M_cam_from_uv @ uv.T).T
        # # Depth type = z - Normalize such that |z| = 1
        # self.xyz = rearrange(xyz / torch.abs(xyz[:, 2:3]), "(h w) c -> h w c", h=self.H_orig).cuda()
        # Depth type = distance - Normalize such that ||ray||=1
        self.xyz = rearrange(F.normalize(xyz, p=2, dim=-1), "(h w) c -> h w c", h=self.H_orig).cuda()

        image_coords = torch.meshgrid(torch.arange(self.H_orig), torch.arange(self.W_orig), indexing="ij")
        self.image_coords = (torch.stack(image_coords, dim=-1) + 0.5).cuda()               # (H, W, 2)

    def normalize(self, x):
        """Normalization helper function."""
        return x / torch.norm(x, dim=-1, keepdim=True)

    def viewmatrix(self, lookdir: TensorType, up: TensorType,
                   position: TensorType, subtract_position: bool = False):
        """Construct lookat view matrix."""
        vec2 = self.normalize((lookdir - position) if subtract_position else lookdir)
        vec0 = self.normalize(torch.cross(up, vec2))
        vec1 = self.normalize(torch.cross(vec2, vec0))
        m = torch.stack([vec0, vec1, vec2, position], dim=1)
        return m

    def sparf_pose_interpolation(self, pose1: TensorType["num_samples", 4, 4],
                            pose2: TensorType["num_samples", 4, 4]) -> TensorType[4, 4]:
        # Poses are in Right-Up-Back (Cam 2 World) format
        origin1 = pose1[0, :3, 3]    # Assuming all samples from same camera/pose
        up1 = pose1[0, :3, 1]
        z1 = - pose1[0, :3, 2]
        lookat1 = origin1 + z1
        origin2 = pose2[0, :3, 3]
        up2 = pose2[0, :3, 1]

        ratio = random.uniform(0.0, 1.0)
        new_origin = ratio * origin1 + (1 - ratio) * origin2
        new_up = ratio * up1 + (1 - ratio) * up2
        new_lookat = lookat1
        new_z_axes = new_lookat - new_origin

        new_pose = self.viewmatrix(new_z_axes, new_up, new_origin)
        new_pose = torch.concat([new_pose, torch.zeros((1, 4), device=self.device)], axis=0)
        new_pose[3, 3] = 1.0
        
        # Poses are in Left-Up-Front (Cam to World) format, convert it to Right-Up-Back (Cam to World) format
        new_pose[:, 0] = - new_pose[:, 0]
        new_pose[:, 2] = - new_pose[:, 2]
        return new_pose.float()
    
    def invert_transform(self, transform: TensorType[3, 4]) -> TensorType[3, 4]:
        R = transform[:3, :3]
        t = transform[:3, 3]
        R_inv = R.T
        t_inv = -R.T @ t
        return torch.concat([R_inv, t_inv.unsqueeze(1)], axis=1)

    def warp_rgbd_to_new_pose(self, ray_indices: TensorType, src_distance: TensorType,
            src_pose: TensorType, dst_pose: TensorType) -> Tuple[TensorType, TensorType]:
        """Warp depth-map from src_pose to dst_pose, at indices of src_pose image"""
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x]
        y_idx = coords[..., 0].long()  # (num_rays,) get rid of the last dimension
        x_idx = coords[..., 1].long()  # (num_rays,) get rid of the last dimension

        z_xyz = self.xyz[..., 2]
        y_xyz = self.xyz[..., 1]
        x_xyz = self.xyz[..., 0]
        z = z_xyz[y_idx, x_idx]
        y = y_xyz[y_idx, x_idx]
        x = x_xyz[y_idx, x_idx]
        coords = torch.stack([x, y, z], -1).cuda()     # (N, 3)

        src_pose = src_pose.cuda()
        dst_pose = dst_pose.cuda()
        R_src = src_pose[0, :3, :3]         # (3, 3) in Right-Up-Back (Cam 2 World) format
        t_src = src_pose[0, :3, 3]          # (3, 1) in Right-Up-Back (Cam 2 World) format
        dst_pose_inv = self.invert_transform(dst_pose)
        R_dst = dst_pose_inv[:3, :3]        # (3, 3) in Right-Up-Back (World 2 Cam) format
        t_dst = dst_pose_inv[:3, 3]         # (3, 1) in Right-Up-Back (World 2 Cam) format

        # src_depth = src_distance.squeeze() * self.distance_per_z[y_idx, x_idx]   # (N)
        # P_src = coords * src_depth.unsqueeze(1)         # (N, 3)
        P_src = coords * src_distance                   # (N, 3)
        P_world = (R_src @ P_src.T + t_src[:, None])
        P_dst = (R_dst @ P_world + t_dst[:, None])      # (3, N)
        P_dst = torch.cat([P_dst, torch.ones((1, P_dst.shape[1]), device=self.device)], dim=0)
        z_depth = - P_dst[2, :]                         # (N,)

        ndc_dst = self.M_ndc_from_cam @ P_dst           # (4, N)
        ndc_dst = ndc_dst / (ndc_dst[3, :] + 1e-8)      # (4, N)
        uv_dst = self.M_uv_from_ndc @ ndc_dst           # (4, N)
        uv_dst = uv_dst[:2, :]                          # (2, N)

        valid_pixels = (uv_dst[0, :] >= 0) & (uv_dst[0, :] < self.W_orig) &\
                       (uv_dst[1, :] >= 0) & (uv_dst[1, :] < self.H_orig) &\
                       (z_depth > 0)
        # Return indices as (cam(0), row, col)
        return torch.stack([torch.zeros(uv_dst.shape[1], device=self.device),
                            uv_dst[1, :], uv_dst[0, :]], dim=1), valid_pixels, z_depth
    
    def ray_generator(self, indices: TensorType, pose: TensorType) -> RayBundle:
        """ Create a camera at pose, and use indices to create RayBundle object"""
        camera = Cameras(fx=self.metadata["fx"], fy=self.metadata["fy"],
                         cx=self.metadata["cx"], cy=self.metadata["cy"],
                         height=self.H_orig, width=self.W_orig,
                         camera_to_worlds=pose[:3, :4].unsqueeze(0),
                         camera_type=CameraType.PERSPECTIVE,
                         M_cam_from_uv=self.M_cam_from_uv.unsqueeze(0)).to(self.device)
        cam_pose_optim = CameraOptimizerConfig().setup(num_cameras=1, device=self.device)
        ray_generator = RayGenerator(camera, cam_pose_optim)
        ray_bundle = ray_generator(indices)
        ray_bundle.nears = torch.ones((indices.shape[0], 1), device=self.device) * self.config.near_plane
        ray_bundle.fars = torch.ones((indices.shape[0], 1), device=self.device) * self.config.far_plane
        return ray_bundle