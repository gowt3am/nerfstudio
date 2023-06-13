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

import h5py
import warnings
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageFilter
from typing import List
from torchtyping import TensorType

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (PerspectiveCameras, PointsRasterizer,
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
                 labels: List[str] = [], test_tuning: bool = False, pregen_random_views: bool = False,
                 on_the_fly_random_views: bool = False):
        super().__init__(dataparser_outputs, scale_factor)
        self.labels = labels
        self.test_tuning = test_tuning
        self.pregen_random_views = pregen_random_views
        self.on_the_fly_random_views = on_the_fly_random_views
        
        self.img_filenames = dataparser_outputs.image_filenames
        self.depth_filenames = self.metadata["depth_filenames"]
        self.normal_filenames = self.metadata["normal_filenames"]
        self.semantic_filenames = self.metadata["semantic_filenames"]
        self.semantic_instance_filenames = self.metadata["semantic_instance_filenames"]
        self.entity_id_filenames = self.metadata["entity_id_filenames"]
        self.gen_image_filenames = self.metadata["gen_image_filenames"]
        self.gen_mask_filenames = self.metadata["gen_mask_filenames"]
        self.gen_poses = self.metadata["gen_poses"]

        self.m_per_asset_unit = self.metadata["m_per_asset_unit"]
        self.H_orig = self.metadata["H_orig"]
        self.W_orig = self.metadata["W_orig"]
        self.H = int(self.H_orig * scale_factor)
        self.W = int(self.W_orig * scale_factor)
        self.xyz_min = self.metadata["xyz_min"]
        self.xyz_max = self.metadata["xyz_max"]
        self.scene_boundary = self.metadata["scene_boundary"]
        self.M_cam_from_uv = self.metadata["M_cam_from_uv"]
        self.fx = self.metadata["fx"]
        self.fy = self.metadata["fy"]
        self.cx = self.metadata["cx"]
        self.cy = self.metadata["cy"]
        self.orig_poses = self.metadata["orig_poses"]

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
            else:
                raise NotImplementedError(f"Label {label} is not implemented")
            
            assert content is not None, f"Content for label {label} is unavailable"
            if label == "pose":
                metadata[label] = content
            elif label == "depth":
                metadata[label] = self._downscale_content(content, label)
            else:
                metadata[label] = self._downscale_content(torch.from_numpy(content), label)
        return metadata
    
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
            self.P_world = hypersim_generate_pointcloud(self.ray_centers_cam, self.orig_poses, all_depths)
            self.all_depths = hypersim_clip_depths_to_bbox(depths=all_depths, 
                P_world=self.P_world, poses=self.orig_poses, xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        else:
            self.all_depths = all_depths
        # Scale depth with respect to scene scaling factor (calculated in dataparser)
        self.all_depths *= self._dataparser_outputs.dataparser_scale

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
    
    def prepare_random_view_generation(self) -> None:
        self.use_max_filtering = False
        self.try_num_train_views_per_new_view = 1
        self.f = torch.tensor([self.fx, self.fy], dtype=torch.float32).cuda()
        self.p = torch.tensor([self.cx, self.cy], dtype=torch.float32).cuda()
        K = torch.tensor([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=torch.float32).cuda()
        Kinv = torch.inverse(K)

        uv_2 = np.stack(np.meshgrid(np.arange(0, self.W), np.arange(0, self.H)[::-1]), -1)     # (H, W, 2)
        uv_2 = rearrange(uv_2, "h w c -> (h w) c")
        uv_2 = np.concatenate([uv_2, np.ones((uv_2.shape[0], 1))], axis=-1).astype(np.float32) # (H*W, 3)
        uv_2 = torch.as_tensor(uv_2, dtype=torch.float32).cuda()

        self.all_points = {}
        self.all_colors = {}
        for j in range(len(self.img_filenames)):
            train_data = self.__getitem__(j)
            img2 = train_data["image"]

            D2 = train_data["depth"].cuda()         # (H, W)
            D2 = rearrange(D2, "h w -> (h w) 1")
            R2 = train_data["pose"][:3, :3]         # (3, 3) in Right-Up-Back (Cam 2 World) format
            t2 = train_data["pose"][:3, 3]          # (3, 1) in Right-Up-Back (Cam 2 World) format

            P_2 = D2.T * (Kinv @ uv_2.T)            # (3, H*W)
            P_2[2, :] *= -1
            P_world = (R2 @ P_2 + t2[:, None])

            # Creating batches (bs) of pointclouds in PyTorch3D (here bs = 1)
            P_world_flat = rearrange(P_world, "c hw -> 1 hw c")
            img2_flat = rearrange(img2, "h w c -> 1 (h w) c")
            self.all_points[2*j] = P_world_flat
            self.all_colors[2*j] = img2_flat

            # White mask for knowing invalid points (and filtering only them)
            self.all_points[2*j + 1] = P_world_flat
            self.all_colors[2*j + 1] = torch.ones_like(img2_flat)
        self.rand_indices = [x + len(self.img_filenames) for x in np.random.choice(self.num_random_views, 50, replace=False)] 
        self.rand_poses = self.gen_poses[self.rand_indices]
    
    def generate_random_views(self, num_views: int) -> Dict:
        """Generate random views of gen_poses on the fly from training images"""
        # Randomly sample num_views random poses from gen_poses
        self.rand_indices = [x + len(self.img_filenames) for x in np.random.choice(self.num_random_views, num_views, replace=False)]
        
        self.rand_poses = self.gen_poses[self.rand_indices]
        self.rendered_images = []
        self.rendered_masks = []
        self.rand_indices_dict = {}
        for i in range(num_views):
            tgt_pose = self.rand_poses[i]
            # Transform from Right-Up-Back (Cam to World) format to
            # Left-Up-Front (World to Cam) format of PyTorch3D
            R_gt = tgt_pose[:3,:3].cpu().numpy().copy()
            t_gt = tgt_pose[:3, 3].cpu().numpy().copy()
            R_gt[:, 0] = -R_gt[:, 0]
            R_gt[:, 2] = -R_gt[:, 2]
            t_gt = torch.as_tensor(-R_gt.T @ t_gt[:, None], dtype=torch.float32).cuda().squeeze()
            R_gt = torch.as_tensor(R_gt.T, dtype=torch.float32).cuda()

            # Randomly sampling few closest and few random images from trainset for each new random view
            # t_gt_np = tgt_pose[:3, 3].numpy()
            # distances_to_gt = []
            # for idx in range(len(self.img_filenames)):
            #     data = self.__getitem__(idx)
            #     cam1_to_world = data["pose"].numpy()
            #     t1 = cam1_to_world[:3, 3]
            #     distances_to_gt.append(np.linalg.norm(t1 - t_gt_np))
            # distances_to_gt = np.array(distances_to_gt)
            # closest_images = np.argsort(distances_to_gt)[:self.NUM_CLOSEST_TRAIN_VIEWS]
            # random_images = np.random.choice(np.delete(np.arange(len(self.img_filenames)), \
            #                     closest_images), self.NUM_RANDOM_TRAIN_VIEWS, replace=False)
            # closest_images = np.concatenate([closest_images, random_images])

            # Randomly sample few training images
            train_idxs = np.random.choice(len(self.img_filenames), self.try_num_train_views_per_new_view, replace=False)

            points = []
            colors = []
            for idx in train_idxs:
                points.append(self.all_points[2*idx])
                colors.append(self.all_colors[2*idx])
                points.append(self.all_points[2*idx + 1])
                colors.append(self.all_colors[2*idx + 1])
            points = torch.cat(points, dim=0).cuda().float()
            colors = torch.cat(colors, dim=0).cuda().float()
            point_cloud = Pointclouds(points=points, features=colors)

            # Creating batches of cameras (here bs = 1)
            # Technically the R, t are from whatever frame each pointcloud is in, to the rendering frame.
            # Additionally R needs to be transposed (row-major format in Pytorch3D)
            cameras = PerspectiveCameras(focal_length=self.f.unsqueeze(0), principal_point=self.p.unsqueeze(0),
                                        image_size=[(self.H, self.W)], R=R_gt.T.unsqueeze(0), T=t_gt.unsqueeze(0),
                                        in_ndc=False, device=torch.device("cuda:0"))
            raster_settings = PointsRasterizationSettings(image_size=(self.H, self.W), radius=1.0/min(self.H, self.W)*2.0, points_per_pixel=2)
            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
            renderer = PointsRenderer(rasterizer=rasterizer, compositor=NormWeightedCompositor())
            rendered_images = renderer(point_cloud).cpu().numpy()

            # Finding maximum coverage of image among all rendered images
            if self.try_num_train_views_per_new_view > 1:
                overlap_area = []
                for idx in range(0, 2*self.try_num_train_views_per_new_view, 2):
                    render_mask = (rendered_images[idx+1, :, :, :]*255).astype(np.uint8)
                    overlap_area.append(np.sum(np.all(render_mask  != 0, axis=-1)))
                max_overlap_idx = np.argmax(overlap_area)
            else:
                max_overlap_idx = 0
            image = (rendered_images[max_overlap_idx*2, :, :, :]*255).astype(np.uint8)
            mask = (rendered_images[max_overlap_idx*2+1, :, :, :]*255).astype(np.uint8)

            # Apply max filter twice to remove missing pixels (ignore actual black objects)
            if self.use_max_filtering:
                img_0 = Image.fromarray(image)
                img_1 = np.array(img_0.filter(ImageFilter.MaxFilter(3)))
                black_pixels = np.where(np.all(mask == 0, axis=-1))
                img_2 = image.copy()
                img_2[black_pixels] = img_1[black_pixels]
                changed_pixels = np.where(np.all(image != img_2, axis=-1))
                mask[changed_pixels] = np.array([255, 255, 255])

                img_3 = Image.fromarray(img_2)
                img_4 = np.array(img_3.filter(ImageFilter.MaxFilter(3)))
                black_pixels = np.where(np.all(mask == 0, axis=-1))
                img_5 = img_2.copy()
                img_5[black_pixels] = img_4[black_pixels]
                changed_pixels = np.where(np.all(img_2 != img_5, axis=-1))
                mask[changed_pixels] = np.array([255, 255, 255])
            else:
                img_5 = image

            self.rendered_images.append(img_5)
            self.rendered_masks.append(mask[:,:,0])
            self.rand_indices_dict[self.rand_indices[i]] = i

        # If original image, then set its index to -1
        for j in range(len(self.image_filenames)):
            self.rand_indices_dict[j] = -1
        return self.rand_indices_dict
        
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