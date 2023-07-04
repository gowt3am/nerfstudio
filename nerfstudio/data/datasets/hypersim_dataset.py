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
HyperSim dataset.
"""

import cv2
import h5py
import warnings
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageFilter
from typing import List, Tuple
from torchtyping import TensorType

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (FoVPerspectiveCameras, PointsRasterizer,
    PointsRasterizationSettings, NormWeightedCompositor, PointsRenderer)

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import hypersim_generate_camera_rays, \
                        hypersim_generate_pointcloud, hypersim_clip_depths_to_bbox

class HyperSimDataset(InputDataset):
    """Wrapper for HyperSim dataset that loads actual images, labels

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The downscaling factor for the dataparser outputs.
    """
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0,
                 labels: List[str] = [], **kwargs):
        super().__init__(dataparser_outputs, scale_factor)
        self.labels = labels
        self.test_tuning = kwargs.get("test_tuning", False)
        self.pregen_random_views = kwargs.get("pregen_random_views", False)
        self.on_the_fly_random_views = kwargs.get("on_the_fly_random_views", False)
        self.rendered_depth_new_views = kwargs.get("rendered_depth_new_views", False)
        self.dk_mask = kwargs.get("dilation_mask", 2)
        self.dk_edge = kwargs.get("dilation_edge", 3)

        self.img_filenames = dataparser_outputs.image_filenames
        self.depth_filenames = self.metadata["depth_filenames"]
        self.normal_filenames = self.metadata["normal_filenames"]
        self.semantic_filenames = self.metadata["semantic_filenames"]
        self.semantic_instance_filenames = self.metadata["semantic_instance_filenames"]
        self.entity_id_filenames = self.metadata["entity_id_filenames"]
        self.gen_image_filenames = self.metadata["gen_image_filenames"]
        self.gen_mask_filenames = self.metadata["gen_mask_filenames"]
        self.gen_poses = self.metadata["gen_poses"]
        self.render_camera_idx = self.metadata["nearest_cam_ids"]
        self.ordered_gen_poses = self.metadata["ordered_gen_poses"]
        
        self.m_per_asset_unit = self.metadata["m_per_asset_unit"]
        self.H_orig = self.metadata["H_orig"]
        self.W_orig = self.metadata["W_orig"]
        self.H = int(self.H_orig * scale_factor)
        self.W = int(self.W_orig * scale_factor)
        self.xyz_min = self.metadata["xyz_min"]
        self.xyz_max = self.metadata["xyz_max"]
        self.scene_boundary = self.metadata["scene_boundary"]
        self.M_cam_from_uv = self.metadata["M_cam_from_uv"].cpu()
        self.M_ndc_from_cam = self.metadata["M_ndc_from_cam"]
        self.M_uv_from_ndc = self.metadata["M_uv_from_ndc"]
        self.M_warp_cam_pts = self.metadata["M_warp_cam_pts"]
        self.M_p3dcam_from_cam = self.metadata["M_p3dcam_from_cam"]
        self.fov_rad_y = self.metadata["fov_rad_y"]
        self.fx = self.metadata["fx"]
        self.fy = self.metadata["fy"]
        self.cx = self.metadata["cx"]
        self.cy = self.metadata["cy"]
        self.orig_poses = self.metadata["orig_poses"]

        self.metadata["distance_per_z"] = torch.from_numpy(self._distance_per_z())
        if "depth" in self.labels:
            self._process_and_clip_depth()

        if self.test_tuning:
            print(f"Test Tuning is enabled, so loading the reconstructed test-view images and their masks")
            self.labels = labels + ["pregenerated", "mask"]
        elif self.pregen_random_views:
            self.num_random_views = len(self.gen_image_filenames) - len(self.img_filenames)
            print(f"Pregenerated Random Views is enabled, so loading {self.num_random_views} additional views for training")
            self.labels = labels + ["mask"]
        elif self.on_the_fly_random_views:
            print(f"On the fly Random Views is enabled, so generating random views every epoch")
            self.num_random_views = self.gen_poses.shape[0] - len(self.img_filenames)
            self.labels = labels + ["mask", "pose"]
            self.prepare_random_view_generation()
        elif self.rendered_depth_new_views:
            print(f"Rendering new views from rendered depth is enabled")
            self.labels = labels + ["pose", "closest_pose"]
            self.find_closest_poses()

    def get_image(self, image_idx: int) -> TensorType["image_height", "image_width", "num_channels"]:
        """Load image, apply tonemapping, rescale it and return 3-channel float32 tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self.img_filenames[image_idx]
        hdr_image = h5py.File(image_filename, 'r')['dataset'][:].astype('float32')
        render_entity_id_filename = self.entity_id_filenames[image_idx]
        render_entity_id = h5py.File(render_entity_id_filename, 'r')['dataset'][:].astype('int32')
        image = self._tonemapping(hdr_image, render_entity_id)

        # rescaled shape should be (h, w, 3), dtype is float32, range is [0, 1]
        image = self._downscale_content(torch.from_numpy(image.astype('float32')), "color")
        assert len(image.shape) == 3
        assert image.shape[2] == 3, f"Image shape of {image.shape} is incorrect."
        return image

    def _tonemapping(self, hdr_image: np.ndarray, render_entity_id: np.ndarray) -> np.ndarray:
        """From https://github.com/apple/ml-hypersim/blob/main/code/python/tools/scene_generate_images_tonemap.py
                
        Compute brightness according to "CCIR601 YIQ" method, use CGIntrinsics strategy for tonemapping, see
        [1] https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py
        [2] https://landofinterruptions.co.uk/manyshades
        """
        assert (render_entity_id != 0).all()
        gamma                             = 1.0/2.2 # standard gamma correction exponent
        inv_gamma                         = 1.0/gamma
        percentile                        = 90 # want this percentile brightness value in unmodified image ...
        brightness_nth_percentile_desired = 0.8 # ... to be this bright after scaling
        valid_mask = render_entity_id != -1
        
        if np.count_nonzero(valid_mask) == 0:
            # If there are no valid pixels, set scale to 1.0
            scale = 1.0 
        else:
            # "CCIR601 YIQ" method for computing brightness
            brightness = 0.3 * hdr_image[:,:,0] + 0.59 * hdr_image[:,:,1] + 0.11 * hdr_image[:,:,2] 
            brightness_valid = brightness[valid_mask]

            # If the kth percentile brightness value in the unmodified image is 
            # less than this, set the scale to 0.0 to avoid divide-by-zero
            eps                               = 0.0001
            brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

            if brightness_nth_percentile_current < eps:
                scale = 0.0
            else:
                # Snavely uses the following expression in the code at 
                # https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
                # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma 
                #         - np.log(brightness_nth_percentile_current))
                #
                # Our expression below is equivalent, but is more intuitive, 
                # because it follows more directly from the expression:
                # (scale*brightness_nth_percentile_current)^gamma = 
                #           = brightness_nth_percentile_desired
                scale = np.power(brightness_nth_percentile_desired, inv_gamma) 
                scale /= brightness_nth_percentile_current
        return np.clip(np.power(np.maximum(scale * hdr_image, 0), gamma), 0, 1)

    def _downscale_content(self, content: TensorType, label: str = "color") -> TensorType:
        if self.scale_factor != 1.0:
            height, width = content.shape[:2]
            self.new_size = (int(height * self.scale_factor), int(width * self.scale_factor))
            if label == 'color':
                content = F.interpolate(content.permute(2,0,1), size = self.new_size,
                                        mode='bilinear').permute(1,2,0)
            elif label == 'depth':
                # Nearest neighbor instead of bilinear, in order to retain missing elements
                content =  F.interpolate(torch.unsqueeze(content, dim = 0),
                                         size = self.new_size, mode='nearest').squeeze(0)
            elif label in ['normals', 'normals_depth']:
                # Nearest neighbor instead of bilinear, in order to retain missing elements
                content = F.interpolate(content.permute(2,0,1), size = self.new_size,
                                        mode='nearest').permute(1,2,0)
                # Normalize normals to unit length, because
                # unit length gets ruined after interpolation.
                # But keep the zero vector missing element code.
                with torch.no_grad():
                    zero_idx = (content.abs().sum(-1) != 0.0).unsqueeze(-1)
                    norm = F.normalize(content, p = 2.0, dim = -1)
                    content = torch.where(zero_idx, content, norm)
            elif label == 'semantics':
                content = F.interpolate(torch.unsqueeze(content, dim = 0).float(),
                    size = self.new_size, mode='nearest').squeeze(0).type(content.dtype)
            elif label == 'pregenerated':
                content = F.interpolate(content.permute(2,0,1), size = self.new_size,
                                        mode='bilinear').permute(1,2,0)
            elif label == 'mask':
                content = F.interpolate(torch.unsqueeze(content, dim = 0).float(),
                    size = self.new_size, mode='nearest').squeeze(0).type(content.dtype)
        return content

    def get_metadata(self, data: Dict) -> Dict:
        """Method used to load additional labels for requested data["image_idx"]

        Args:
            image_idx: The image index in the dataset.
        """
        metadata = {}
        image_idx = data["image_idx"]
        for label in self.labels:
            if label == 'normals_depth':
                # TODO : If this is required, use it once the model code is done
                raise NotImplementedError("Normals from depth as labels is not implemented")
            
            elif label == "depth":
                content =  self.all_depths[image_idx]

            elif label == "normals":
                content = h5py.File(self.normal_filenames[image_idx], 'r')['dataset'][:].astype('float32')
                if content is None:
                    content = np.zeros((self.H_orig, self.W_orig, 3), dtype=np.float32)
                zero_vect = np.zeros(3, dtype=content.dtype)
                content[np.isnan(np.abs(content).sum(axis=-1))] = zero_vect
            
            elif label in ["semantics", "semantics_WF"]:
                content = h5py.File(self.semantic_filenames, 'r')['dataset'][:].astype('int64')
                if content is None:
                    content = -1 * np.ones((self.H_orig, self.W_orig), dtype=np.int64)
                content[content == -1] = 0

                # Check all semantic classes in scene and rearrange class labels
                class_ids = np.unique(content).astype(np.uint8)
                if label == "semantics":
                    # New class IDs go from [0, C)
                    for old_id in class_ids:
                        content[content == old_id] = old_id - 1
                else:
                    # Merge Window to Wall class
                    content[content == 9] = 1
                    # Merge Mirror to Wall class
                    # content[content == 19] = 1
                    # Merge FloorMat into Floor class
                    content[content == 20] = 2
                    wall_floor_mask = (content == 1) + (content == 2)
                    content[np.logical_not(wall_floor_mask)] = 3

            elif label == "pregenerated":
                content = np.array(Image.open(self.gen_image_filenames[image_idx],
                                              'r')).astype('float32') / 255.0
                if content is None:
                    content = np.zeros((self.H_orig, self.W_orig, 3), dtype=np.float32)
                zero_vect = np.zeros(3, dtype=content.dtype)
                content[np.isnan(np.abs(content).sum(axis=-1))] = zero_vect

            elif label == "mask":
                if self.test_tuning:
                    # Only in this case, we load pregen_masks here
                    content = np.array(Image.open(self.gen_mask_filenames[image_idx],
                                                  'r')).astype('float32') / 255.0
                else:
                    # In all other cases, its original train_images so use full white mask
                    content = None
                if content is None:
                    content = np.ones((self.H_orig, self.W_orig), dtype=np.float32)
                content[np.isnan(np.abs(content).sum(axis=-1))] = 1.0
            elif label == "pose":
                content = self.gen_poses[image_idx]
            elif label == "closest_pose":
                content = self.closest_poses[image_idx]
            else:
                raise NotImplementedError(f"Label {label} is not implemented")
            
            assert content is not None, f"Content for label {label} is unavailable"
            if label == "pose" or label == "closest_pose":
                metadata[label] = content
            elif label == "depth":
                metadata[label] = self._downscale_content(content, label)
            else:
                metadata[label] = self._downscale_content(torch.from_numpy(content), label)
        return metadata
    
    def _distance_per_z(self):
        '''Division factor for converting depth from a distance to camera center to z-plane value
        Code taken from: https://github.com/apple/ml-hypersim/issues/9#issuecomment-754935697
        '''
        w_lim_min = (-0.5 * self.W_orig) + 0.5
        w_lim_max = (0.5 * self.W_orig) - 0.5
        im_plane_X = np.linspace(w_lim_min, w_lim_max, self.W_orig)
        im_plane_X = im_plane_X.reshape(1, self.W_orig).repeat(self.H_orig, 0)
        im_plane_X = im_plane_X.astype(np.float32)[:, :, None]
        h_lim_min = (-0.5 * self.H_orig) + 0.5
        h_lim_max = (0.5 * self.H_orig) - 0.5
        im_plane_Y = np.linspace(h_lim_min, h_lim_max, self.H_orig)
        im_plane_Y = im_plane_Y.reshape(self.H_orig, 1).repeat(self.W_orig, 1)
        im_plane_Y = im_plane_Y.astype(np.float32)[:, :, None]
        im_plane_Z = np.full([self.H_orig, self.W_orig, 1], self.fx, np.float32)
        im_plane = np.concatenate([im_plane_X, im_plane_Y, im_plane_Z], 2)
        im_plane_norm2_inv = 1.0 / np.linalg.norm(im_plane, 2, 2)
        return im_plane_norm2_inv * self.fx

    def _process_and_clip_depth(self):
        """Preload all depth files, compute full pointcloud, and crop depth values to new scene bounds if needed"""
        print(f"Preloading all depths, to get pointcloud and crop depth values to new scene bounds...")
        all_depths = []
        for file in self.depth_filenames:
            all_depths.append(h5py.File(file, 'r')['dataset'][:].astype('float32'))
            if all_depths[-1] is None:
                all_depths[-1] = np.zeros((self.H_orig, self.W_orig), dtype=np.float32)
        all_depths = np.asarray(all_depths)
        all_depths[np.isnan(all_depths)] = 0.0
        all_depths = torch.from_numpy(all_depths)

        # Converting depth from meters to asset units
        all_depths /= self.m_per_asset_unit 
        if (self.xyz_min != self.scene_boundary['xyz_scene_min']).any() or \
           (self.xyz_max != self.scene_boundary['xyz_scene_max']).any():
            self.ray_centers_cam = hypersim_generate_camera_rays((self.H, self.W), self.M_cam_from_uv)
            self.P_world = hypersim_generate_pointcloud(self.ray_centers_cam, self.orig_poses,
                                                        all_depths, "distance")
            self.all_depths = hypersim_clip_depths_to_bbox(depths=all_depths, 
                P_world=self.P_world, poses=self.orig_poses, xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        else:
            self.all_depths = all_depths
        # Scale depth with respect to scene scaling factor (calculated in dataparser)
        self.all_depths *= self._dataparser_outputs.dataparser_scale
    
    def prepare_random_view_generation(self) -> None:
        u = np.linspace(-1.0 + 1.0/self.W_orig, 1.0 - 1.0/self.W_orig, self.W_orig)
        v = np.linspace(-1.0 + 1.0/self.H_orig, 1.0 - 1.0/self.H_orig, self.H_orig)
        # Reverse vertical coordinate because [H=0, W=0] corresponds to (u=-1, v=1)
        u, v = np.meshgrid(u, v[::-1])
        uv_2 = torch.as_tensor(np.dstack((u, v, np.ones_like(u))), dtype=torch.float32)    # (H, W, 3)
        uv_2 = rearrange(uv_2, 'h w c -> (h w) c')

        xyz_2 = (self.M_cam_from_uv @ uv_2.T).T
        # Depth type = z - Normalize such that |z| = 1
        # self.xyz_2 = xyz_2 / torch.abs(xyz_2[:, 2:3])
        # # Depth type = distance - Normalize such that ||ray||=1
        xyz_2 = F.normalize(xyz_2, p=2, dim=-1)

        self.all_points = {}
        self.all_colors = {}
        for j in range(len(self.img_filenames)):
            train_data = self.__getitem__(j)
            img2 = train_data["image"]

            D2 = train_data["depth"].clone().cpu()
            D2 = rearrange(D2, "h w -> (h w) 1")
            P_2 = xyz_2 * D2                        # (H*W, 3)

            R2 = train_data["pose"][:3, :3].cpu()   # (3, 3) in Right-Up-Back (Cam 2 World) format
            t2 = train_data["pose"][:3, 3].cpu()    # (3, 1) in Right-Up-Back (Cam 2 World) format
            P_world = (R2 @ P_2.T + t2[:, None])    # (3, H*W)
    
            P_world_flat = rearrange(P_world, "c hw -> hw c")
            img2_flat = rearrange(img2, "h w c -> 1 (h w) c")
            self.all_points[2*j] = P_world_flat
            self.all_colors[2*j] = img2_flat

            # White mask for knowing invalid points (and filtering only them)
            # self.all_points[2*j + 1] = self.all_points[2*j]
            self.all_colors[2*j + 1] = torch.ones_like(self.all_colors[2*j])
        
        self.rand_indices = [x + len(self.img_filenames) for x in \
                             np.random.choice(self.num_random_views, 50, replace=False)] 
        self.rand_poses = self.gen_poses[self.rand_indices]
    
    def find_closest_poses(self):
        """For all the camera views in training set, find the closest camera view"""
        closest_poses = []
        for i in range(self.gen_poses.shape[0]):
            tgt_pose = self.gen_poses[i]
            t_gt_np = tgt_pose[:3, 3].numpy()
            distances_to_gt = []
            for idx in range(self.gen_poses.shape[0]):
                if idx != i:
                    t1 = self.gen_poses[idx].numpy()[:3, 3]
                    distances_to_gt.append(np.linalg.norm(t1 - t_gt_np))
                else:
                    distances_to_gt.append(np.inf)
            distances_to_gt = np.array(distances_to_gt)
            closest_idx = np.argsort(distances_to_gt)[0]
            closest_poses.append(self.gen_poses[closest_idx])
        self.closest_poses = torch.stack(closest_poses, dim=0)

    def detect_edges(self, image: np.ndarray, valid_mask: np.ndarray, dk_mask: int = 2,
                     dk_edge: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Detect edges in image, ignore the invalid mask regions, and dilate the edges"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = np.zeros_like(edges)
        edge_mask[edges > 0] = 255
        
        invalid_mask = (valid_mask == 0).astype(np.uint8)
        invalid_mask = cv2.dilate(invalid_mask, np.ones((dk_mask, dk_mask)), iterations=1)
        edge_mask[invalid_mask > 0] = 0

        if dk_edge > 0:
            edge_mask = cv2.dilate(edge_mask, np.ones((dk_edge, dk_edge)), iterations=1)

        # Set pixels within the dilated mask to black
        result = image.copy()
        result[edge_mask > 0] = 0
        return result, edge_mask

    def generate_random_views(self, num_views: int, epoch: int) -> Dict:
        """Generate random views of gen_poses on the fly from training images"""
        if self.ordered_gen_poses > 0:
            # Use epoch to sample num_views in ordered fashion [Useful for slowly moving away poses]
            self.rand_indices = [x + len(self.img_filenames) for x in
                                np.arange(epoch*num_views, (epoch+1)*num_views, 1)]
        else:
            # Randomly sample num_views random poses from gen_poses
            self.rand_indices = [x + len(self.img_filenames) for x in
                                np.random.choice(self.num_random_views, num_views, replace=False)]
        
        self.rand_poses = self.gen_poses[self.rand_indices]
        self.rendered_images = []
        self.rendered_masks = []
        self.rand_indices_dict = {}
        for i in range(num_views):
            tgt_pose = self.rand_poses[i].cpu().clone()
            
            # Transform from Right-Up-Back (Cam to World) format of Hypersim to
            # Right-Up-Back (World to Cam) format
            R = tgt_pose[:3,:3]
            t = tgt_pose[:3, 3]
            pose = torch.eye(4)
            pose[:3, :3] = R.T
            pose[:3, 3] = -R.T @ t
            
            # Select the nearest training view for each new random view
            train_idx = self.render_camera_idx[self.rand_indices[i]]
            P_world_orig = self.all_points[2*train_idx]
            P_world_orig = torch.cat([P_world_orig, torch.ones_like(P_world_orig[:, :1])], dim=-1)
            P_cam = self.M_p3dcam_from_cam @ self.M_warp_cam_pts @ pose @ P_world_orig.T
            P_cam = P_cam[:3, :].T.unsqueeze(0)

            points = torch.cat([P_cam, P_cam], dim=0)
            colors = torch.cat([self.all_colors[2*train_idx], self.all_colors[2*train_idx + 1]], dim=0)
            point_cloud = Pointclouds(points=points.cuda().float(), features=colors.cuda().float())

            cameras = FoVPerspectiveCameras(device=torch.device("cuda:0"), fov=self.fov_rad_y, degrees=False,
                                            aspect_ratio=1.0, znear=1.0, zfar=1000.0)
            raster_settings = PointsRasterizationSettings(image_size=(self.H, self.W), radius=1.0/min(self.H, self.W)*2.0, points_per_pixel=2)
            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
            renderer = PointsRenderer(rasterizer=rasterizer, compositor=NormWeightedCompositor())
            rendered_images = renderer(point_cloud).cpu().numpy()
            image = (rendered_images[0, :, :, :]*255).astype(np.uint8)
            mask = (rendered_images[1, :, :, 0]*255).astype(np.uint8)

            if self.dk_edge > -1:
                image, edge_mask = self.detect_edges(image, mask, dk_mask=self.dk_mask,
                                                     dk_edge=self.dk_edge)
                mask[edge_mask > 0] = np.array([0, 0, 0])

            save_dest = "/scratch_net/bmicdl02/gsenthi/data/temp/"
            img = Image.fromarray(image)
            img.save(save_dest + f"img_{self.rand_indices[i]}.png")
            
            self.rendered_images.append(image)
            self.rendered_masks.append(mask)
            self.rand_indices_dict[self.rand_indices[i]] = i

        # If original image, then set its index to -1
        for j in range(len(self.image_filenames)):
            self.rand_indices_dict[j] = -1
        return self.rand_indices_dict

    def __get_pregen_rand_item__(self, image_idx: int) -> Dict:
        data = {"image_idx": image_idx}
        data["pose"] = self.gen_poses[image_idx]
        
        image = np.array(Image.open(self.gen_image_filenames[image_idx], 'r')).astype('float32') / 255.0
        if image is None:
            image = np.zeros((self.H_orig, self.W_orig, 3), dtype=np.float32)
        zero_vect = np.zeros(3, dtype=image.dtype)
        image[np.isnan(np.abs(image).sum(axis=-1))] = zero_vect
        data["image"] = self._downscale_content(torch.from_numpy(image), "image")
        
        mask = np.array(Image.open(self.gen_mask_filenames[image_idx], 'r')).astype('float32') / 255.0
        if mask is None:
            mask = np.ones((self.H_orig, self.W_orig), dtype=np.float32)
        mask[np.isnan(np.abs(mask).sum(axis=-1))] = 1.0
        data["mask"] = self._downscale_content(torch.from_numpy(mask), "mask")

        if "depth" in self.labels:
            depth = np.zeros((self.H_orig, self.W_orig), dtype=np.float32)
            depth[np.isnan(depth)] = 0.0
            data["depth"] = self._downscale_content(torch.from_numpy(depth), "depth")
        if "normals" in self.labels:
            normals = np.zeros((self.H_orig, self.W_orig, 3), dtype=np.float32)
            zero_vect = np.zeros(3, dtype=normals.dtype)
            normals[np.isnan(np.abs(normals).sum(axis=-1))] = zero_vect
            data["normals"] = self._downscale_content(torch.from_numpy(normals), "normals")
        return data        

    def __get_on_the_fly_rand_item__(self, image_idx: int) -> Dict:
        relative_idx = self.rand_indices_dict[image_idx]
        data = {"image_idx": image_idx}
        data["pose"] = self.gen_poses[image_idx]
        
        image = (self.rendered_images[relative_idx] / 255.0).astype(np.float32)
        if image is None:
            image = np.zeros((self.H_orig, self.W_orig, 3), dtype=np.float32)
        zero_vect = np.zeros(3, dtype=image.dtype)
        image[np.isnan(np.abs(image).sum(axis=-1))] = zero_vect
        data["image"] = self._downscale_content(torch.from_numpy(image), "image")
        
        mask = (self.rendered_masks[relative_idx] / 255.0).astype(np.float32)
        if mask is None:
            mask = np.ones((self.H_orig, self.W_orig), dtype=np.float32)
        mask[np.isnan(np.abs(mask).sum(axis=-1))] = 1.0
        data["mask"] = self._downscale_content(torch.from_numpy(mask), "mask")

        if "depth" in self.labels:
            depth = np.zeros((self.H_orig, self.W_orig), dtype=np.float32)
            depth[np.isnan(depth)] = 0.0
            data["depth"] = self._downscale_content(torch.from_numpy(depth), "depth")
        if "normals" in self.labels:
            normals = np.zeros((self.H_orig, self.W_orig, 3), dtype=np.float32)
            zero_vect = np.zeros(3, dtype=normals.dtype)
            normals[np.isnan(np.abs(normals).sum(axis=-1))] = zero_vect
            data["normals"] = self._downscale_content(torch.from_numpy(normals), "normals")
        return data
    
    def __get_rand_item__(self, image_idx: int) -> Dict:
        if self.pregen_random_views:
            return self.__get_pregen_rand_item__(image_idx)
        elif self.on_the_fly_random_views:
            return self.__get_on_the_fly_rand_item__(image_idx)
        else:
            raise NotImplementedError("Both Pregen & On the fly Random Views are disabled!")