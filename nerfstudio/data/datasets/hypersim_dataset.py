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
from PIL import Image
from typing import List
from torchtyping import TensorType

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
                 labels: List[str] = [], test_tuning: bool = False, random_views: bool = False):
        super().__init__(dataparser_outputs, scale_factor)
        self.labels = labels
        self.test_tuning = test_tuning
        self.random_views = random_views
        
        self.img_filenames = dataparser_outputs.image_filenames
        self.depth_filenames = self.metadata["depth_filenames"]
        self.normal_filenames = self.metadata["normal_filenames"]
        self.semantic_filenames = self.metadata["semantic_filenames"]
        self.semantic_instance_filenames = self.metadata["semantic_instance_filenames"]
        self.entity_id_filenames = self.metadata["entity_id_filenames"]
        self.reconstructed_image_filenames = self.metadata["reconstructed_image_filenames"]
        self.reconstructed_mask_filenames = self.metadata["reconstructed_mask_filenames"]
        self.reconstructed_poses = self.metadata["reconstructed_poses"]

        if self.test_tuning:
            print("Test Tuning is enabled, so loading the reconstructed images and their masks")
            self.labels = labels + ["reconstructed", "mask"]
        if self.random_views:
            self.num_random_views = len(self.reconstructed_image_filenames) - len(self.img_filenames)
            print(f"Random Views is enabled, so using {self.num_random_views} additional views for training")
            self.labels = labels + ["mask"]

        self.m_per_asset_unit = self.metadata["m_per_asset_unit"]
        self.H_orig = self.metadata["H_orig"]
        self.W_orig = self.metadata["W_orig"]
        self.H = int(self.H_orig * scale_factor)
        self.W = int(self.W_orig * scale_factor)
        self.xyz_min = self.metadata["xyz_min"]
        self.xyz_max = self.metadata["xyz_max"]
        self.scene_boundary = self.metadata["scene_boundary"]
        self.M_cam_from_uv = self.metadata["M_cam_from_uv"]
        self.poses = self.metadata["orig_poses"]

        if "depth" in self.labels:
            self._process_and_clip_depth()

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
            elif label == 'reconstructed':
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

            elif label == "reconstructed":
                content = np.array(Image.open(self.reconstructed_image_filenames[image_idx], 'r')).astype('float32') / 255.0
                if content is None:
                    content = np.zeros((self.H_orig, self.W_orig, 3), dtype=np.float32)
                zero_vect = np.zeros(3, dtype=content.dtype)
                content[np.isnan(np.abs(content).sum(axis=-1))] = zero_vect

            elif label == "mask":
                if self.random_views:
                    content = None
                else:
                    content = np.array(Image.open(self.reconstructed_mask_filenames[image_idx], 'r')).astype('float32') / 255.0

                if content is None:
                    content = np.ones((self.H_orig, self.W_orig), dtype=np.float32)
                content[np.isnan(np.abs(content).sum(axis=-1))] = 1.0
            else:
                raise NotImplementedError(f"Label {label} is not implemented")
            
            assert content is not None, f"Content for label {label} is unavailable"
            metadata[label] = self._downscale_content(torch.from_numpy(content), label) if label != "depth" else self._downscale_content(content, label)
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
            self.P_world = hypersim_generate_pointcloud(self.ray_centers_cam, self.poses, all_depths)
            self.all_depths = hypersim_clip_depths_to_bbox(depths=all_depths, 
                P_world=self.P_world, poses=self.poses, xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        else:
            self.all_depths = all_depths
        # Scale depth with respect to scene scaling factor (calculated in dataparser)
        self.all_depths *= self._dataparser_outputs.dataparser_scale

    def __get_rand_item__(self, image_idx: int) -> Dict:
        data = {"image_idx": image_idx}
        data["pose"] = self.reconstructed_poses[image_idx]
        
        image = np.array(Image.open(self.reconstructed_image_filenames[image_idx], 'r')).astype('float32') / 255.0
        if image is None:
            image = np.zeros((self.H_orig, self.W_orig, 3), dtype=np.float32)
        zero_vect = np.zeros(3, dtype=image.dtype)
        image[np.isnan(np.abs(image).sum(axis=-1))] = zero_vect
        data["image"] = self._downscale_content(torch.from_numpy(image), "image")
        
        mask = np.array(Image.open(self.reconstructed_mask_filenames[image_idx], 'r')).astype('float32') / 255.0
        if mask is None:
            mask = np.ones((self.H_orig, self.W_orig), dtype=np.float32)
        mask[np.isnan(np.abs(mask).sum(axis=-1))] = 1.0
        data["mask"] = self._downscale_content(torch.from_numpy(mask), "mask")

        # if "depth" in self.labels:
        #     depth = np.zeros((self.H_orig, self.W_orig), dtype=np.float32)
        #     depth[np.isnan(depth)] = 0.0
        #     data["depth"] = self._downscale_content(torch.from_numpy(depth), "depth")
        # if "normals" in self.labels:
        #     normals = np.zeros((self.H_orig, self.W_orig, 3), dtype=np.float32)
        #     zero_vect = np.zeros(3, dtype=normals.dtype)
        #     normals[np.isnan(np.abs(normals).sum(axis=-1))] = zero_vect
        #     data["normals"] = self._downscale_content(torch.from_numpy(normals), "normals")
        return data