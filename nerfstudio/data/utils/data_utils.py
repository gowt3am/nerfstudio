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

"""Utility functions to allow easy re-use of common operations across dataloaders"""
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

import torch.nn.functional as F
from einops import rearrange
from torchtyping import TensorType
from nerfstudio.cameras.rays import RayBundle
import faiss

def get_image_mask_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
    if len(mask_tensor.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return mask_tensor


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype="int64").view(1, 1, -1)
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.NEAREST)
    semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [width, height, 1].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float64) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image[:, :, np.newaxis])


""" ############## Complex HyperSim Dataset Utils ############## """
def hypersim_generate_camera_rays(size: Tuple, M_cam_from_uv: TensorType = None) -> Tuple:
    """Create grid of camera rays in camera coordinates"""
    height, width = size

    # Create grid of pixel center positions in the uv space
    u_min  = -1.0; u_max  = 1.0
    v_min  = -1.0; v_max  = 1.0
    half_du = 0.5 * (u_max - u_min) / width
    half_dv = 0.5 * (v_max - v_min) / height
    u_linspace = np.linspace(u_min + half_du, u_max - half_du, width)
    v_linspace = np.linspace(v_min + half_dv, v_max - half_dv, height)
    # Reverse vertical coordinate because [H=0, W=0] corresponds to (u=-1, v=1)
    u, v = np.meshgrid(u_linspace, v_linspace[::-1])
    uvs = torch.FloatTensor(np.dstack((u, v, np.ones_like(u))))
    ray_centers_uv = rearrange(uvs, 'h w xyz-> (h w) xyz')

    # Create grid of pixel center indices in the uv space
    u_idx_linspace = np.arange(0, width)
    v_idx_linspace = np.arange(0, height)
    u_idx, v_idx = np.meshgrid(u_idx_linspace, v_idx_linspace)
    idxs = torch.IntTensor(np.dstack((u_idx, v_idx)))
    ray_idxs = rearrange(idxs, 'h w i-> (h w) i')

    # Transfer pixel center positions from uv to the camera space [HW, 3]
    ray_centers_cam = (M_cam_from_uv @ ray_centers_uv.T).T
    # Normalize such that ||ray_dir||=1
    ray_centers_cam = F.normalize(ray_centers_cam, p = 2, dim = -1)
    return ray_centers_cam, ray_centers_uv, ray_idxs

def hypersim_generate_pointcloud(ray_centers_cam: TensorType,
                                 cam_to_world: TensorType["num_cameras":..., 3, 4],
                                 depths: TensorType) -> TensorType:
    """Generates a pointcloud from depth values for the Hypersim dataset"""
    depth = rearrange(depths.clone(), 'b h w -> b (h w)') 
    if depth.dim() == 2:
        depth = depth.unsqueeze(-1)
    else:
        assert depth.dim() == 3
    # Normalize such that ||ray_dir||=1
    ray_centers_cam = F.normalize(ray_centers_cam.unsqueeze(0), p = 2, dim = -1)
    
    # Converting 3x4 poses into 4x4 poses
    last_row = torch.tensor([0, 0, 0, 1], dtype = cam_to_world.dtype,
                            device = cam_to_world.device).view(1, 1, 4).repeat(cam_to_world.shape[0], 1, 1)
    poses = torch.cat((cam_to_world, last_row), dim = 1)

    P_cam = ray_centers_cam * depth
    P_cam = torch.cat((P_cam, torch.ones_like(depth)), dim = -1)
    P_world = poses @ torch.transpose(P_cam, 1, 2)
    P_world = torch.transpose(P_world, 1, 2)
    return P_world[:, :, :3] / P_world[:, :, 3:4]

def hypersim_clip_depths_to_bbox(depths: TensorType, P_world: TensorType,
                                 poses: TensorType["num_cameras":..., 3, 4],
                                 xyz_min: TensorType, xyz_max: TensorType) -> TensorType:
    h = depths.shape[1]
    depth = rearrange(depths.clone(), 'b h w -> b (h w)')
    assert depth.dim() == 2
    xyz_min = xyz_min.unsqueeze(0).unsqueeze(0)
    xyz_max = xyz_max.unsqueeze(0).unsqueeze(0)

    P_world_bound = P_world.clone()
    P_world_bound = torch.where(P_world > xyz_max, xyz_max, P_world_bound)
    P_world_bound = torch.where(P_world < xyz_min, xyz_min, P_world_bound)
    S = (P_world_bound - poses[:, None, :3, 3]) / (P_world - poses[:, None, :3, 3])
    S = torch.where((depth.unsqueeze(-1) == 0.0), torch.ones_like(S), S)
    S = torch.min(S, dim=-1, keepdim=True)[0]
    S = S[...,0]
    return rearrange(depth * S, 'b (h w) -> b h w', h=h)


@torch.cuda.amp.autocast(dtype=torch.float32)
def hypersim_normals_from_ray_depths(ray_bundle: RayBundle, ray_depth: TensorType) \
                                                -> Tuple[TensorType, TensorType]:
    """Compute normals from sets of triangular rays of the below form
        [P1..., P2..., P3...]

        ____|_P2_|____
        _P3_|_P1_|____
            |    |     
    """
    # Get the corresponding points in world coordinates
    P_world = ray_bundle.origins + ray_bundle.directions * ray_depth
    N, _ = P_world.shape
    assert N % 3 == 0, f"Number of points {N} is not divisible by 3"
    N = N // 3
    P1_world = P_world[:N]
    P2_world = P_world[N:2*N]
    P3_world = P_world[2*N: 3*N]

    # Extract normals as the cross product over the local pixel triangle, and normalize
    # From https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7335535
    normals = torch.cross((P2_world - P1_world), (P3_world - P1_world), dim = -1)
    normals = F.normalize(normals, p = 2.0, dim = -1)

    # TODO: Processing invalid depth and normals
    invalid_depth = (ray_depth == 0.0) + torch.isnan(ray_depth) + torch.isinf(ray_depth)
    P1_invalid = invalid_depth[:N, 0]
    P2_invalid = invalid_depth[N:2*N, 0]
    P3_invalid = invalid_depth[2*N: 3*N, 0]  

    norm_invalid = (torch.abs(normals).sum(-1) == 0.0) + torch.isnan(normals).sum(-1) + torch.isinf(normals).sum(-1)
    invalid_normals = P1_invalid + P2_invalid + P3_invalid + norm_invalid
    normals[invalid_normals] = torch.zeros(3, device=ray_depth.device)

    # Assign the same normal to P1, P2 and P3
    return normals.repeat(3, 1), torch.logical_not(invalid_normals.repeat(3))


def _merge_similar_clusters(similarities, center_idxs, c_idx, old_assignments,
                            merge_threshold, merge):
    """Merge clusters that are similar to the c_idx^th cluster."""
    if merge:
        c_idx_similarities = similarities[c_idx]
        all_c_idx_similar_idxs = center_idxs[c_idx_similarities > merge_threshold]
    else:
        all_c_idx_similar_idxs = torch.tensor([c_idx], dtype=torch.int64, device=similarities.device)
    new_assignments = torch.eq(old_assignments.unsqueeze(1), all_c_idx_similar_idxs.unsqueeze(0)).any(dim=1)
    return all_c_idx_similar_idxs, new_assignments

def _find_opposite(similarities, center_idxs, c_idx, old_assignments,
                   opposite_threshold, merge_threshold, merge):    
    """Find clusters that are opposite to the c_idx^th cluster and merge them."""    
    c_idx_opposite_idx = similarities[c_idx].argmin().item()
    c_idx_opposite_similarity = similarities[c_idx][c_idx_opposite_idx]
    if (-1.0 * c_idx_opposite_similarity) > opposite_threshold:
        _, c_idx_opposite_mask = _merge_similar_clusters(similarities = similarities, 
                    center_idxs = center_idxs, c_idx = c_idx_opposite_idx,
                    old_assignments = old_assignments, merge_threshold = merge_threshold,
                    merge = merge)
        return True, c_idx_opposite_mask
    else:
        return False, None

def _cluster_normals(normals: TensorType, device: torch.device,
        num_clusters: int = 20, num_iterations: int = 20, similar_threshold: float = 0.99, 
        merge_clusters: bool = True, find_opposites = True):
    """Clusters normals using K-means, and merges clusters that are similar"""
    # K-means clustering using faiss library
    # Default min, max per centroids are 39, 256, changed to suppress warnings
    kmeans = faiss.Kmeans(d = 3, k = num_clusters, niter = num_iterations,
                          gpu = False, spherical = True, verbose = False,
                          min_points_per_centroid = 39, max_points_per_centroid = 256)
    kmeans.train(normals)
    
    # Extract cluster membership for each normal
    _, normal_clusters = kmeans.index.search(normals, 1)
    normal_clusters = torch.from_numpy(normal_clusters).to(device).squeeze(1)
    cluster_centers = torch.from_numpy(kmeans.centroids).to(device)
    center_idxs = torch.arange(start = 0, end = cluster_centers.shape[0], step = 1,
                               device=cluster_centers.device)

    # Get largest cluster and assign it as C1
    cluster_sizes = torch.bincount(normal_clusters)
    _, cluster_sizes_sorted_idx = torch.topk(cluster_sizes, cluster_sizes.shape[0],
                                             largest=True, sorted=True)
    c1_idx = cluster_sizes_sorted_idx[0].item()

    # Extracting similarities between clusters
    similarities = torch.matmul(cluster_centers, cluster_centers.T)
    similarities_absolute = torch.abs(similarities)

    # Most orthogonal set (C1, C2, C3)    
    criteria = (similarities_absolute[:, c1_idx].unsqueeze(1) +
                similarities_absolute[c1_idx, :].unsqueeze(0) + similarities_absolute)
    min_clusters, min_cluster_idxs = torch.min(criteria, dim = 0)
    c2_idx = torch.argmin(min_clusters).item()
    c3_idx = min_cluster_idxs[c2_idx].item()

    # Merge clusters that are similar to c_i with c_i
    _, c1_mask = _merge_similar_clusters(similarities = similarities, 
                    center_idxs = center_idxs, c_idx = c1_idx,
                    old_assignments = normal_clusters, merge_threshold = similar_threshold,
                    merge = merge_clusters)
    _, c2_mask = _merge_similar_clusters(similarities = similarities, 
                    center_idxs = center_idxs, c_idx = c2_idx,
                    old_assignments = normal_clusters, merge_threshold = similar_threshold,
                    merge = merge_clusters)
    _, c3_mask = _merge_similar_clusters(similarities = similarities, 
                    center_idxs = center_idxs, c_idx = c3_idx,
                    old_assignments = normal_clusters, merge_threshold = similar_threshold,
                    merge = merge_clusters)
    
    # Assign new clusters
    normal_clusters_new = torch.zeros_like(normal_clusters)
    normal_clusters_new[c1_mask] = 1
    normal_clusters_new[c2_mask] = 2
    normal_clusters_new[c3_mask] = 3
    cluster_centers_new = cluster_centers[[c1_idx, c2_idx, c3_idx]]

    # Find opposite clusters for C1, C2, C3 and merge/assign them
    if find_opposites:
        found_opposite, c1_opposite_mask = _find_opposite(similarities = similarities, 
                                    center_idxs = center_idxs, c_idx = c1_idx, 
                                    old_assignments = normal_clusters,
                                    opposite_threshold = similar_threshold, 
                                    merge_threshold = similar_threshold, 
                                    merge = merge_clusters)
        if found_opposite:
            normal_clusters_new[c1_opposite_mask] = -1
        found_opposite, c2_opposite_mask = _find_opposite(similarities = similarities, 
                                        center_idxs = center_idxs, c_idx = c2_idx, 
                                        old_assignments = normal_clusters,
                                        opposite_threshold = similar_threshold, 
                                        merge_threshold = similar_threshold, 
                                        merge = merge_clusters)
        if found_opposite:
            normal_clusters_new[c2_opposite_mask] = -2
        found_opposite, c3_opposite_mask = _find_opposite(similarities = similarities, 
                                        center_idxs = center_idxs, c_idx = c3_idx, 
                                        old_assignments = normal_clusters,
                                        opposite_threshold = similar_threshold, 
                                        merge_threshold = similar_threshold, 
                                        merge = merge_clusters)
        if found_opposite:
            normal_clusters_new[c3_opposite_mask] = -3
    return normal_clusters_new, cluster_centers_new