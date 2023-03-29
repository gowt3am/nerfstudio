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

"""Data parser for HyperSim dataset"""
import math, json, random, h5py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, List, Dict, Any

import numpy as np
import pandas as pd
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox

random.seed(37)
CAMERA_METADATA_WEBSITE = 'https://raw.githubusercontent.com/apple/ml-hypersim/main/contrib/' + \
                          'mikeroberts3000/metadata_camera_parameters.csv'

@dataclass
class HyperSimDataParserConfig(DataParserConfig):
    """HyperSim dataset config.
    HyperSim dataset (https://github.com/apple/ml-hypersim) is a photorealistic synthetic
    dataset of indoor scenes. This dataparser assumes that one scene of form ai_VVV_NNN was
    extracted to form a scene directory as follows:
     - ../all_scenes_metadata.csv - Pre-generated from ManhattanNeRF project
     - _detail/
        - metadata_abc.csv
        - cam_xx/
            - metadata_abc.hdf5
        - mesh/
            - metadata_abc.hdf5
     - images/
        - scene_cam_xx_final_hdf5/
            - frame.yyyy.color.hdf5
        - scene_cam_xx_geometry_hdf5/
            - frame.yyyy.depth_meters.hdf5
            - frame.yyyy.normal_bump_world.hdf5
            - frame.yyyy.render_entity_id.hdf5
            - frame.yyyy.semantic.hdf5
            - frame.yyyy.semantic_instance.hdf5
    """

    _target: Type = field(default_factory=lambda: HyperSim)
    """target class to instantiate"""
    data: Path = Path("data/hypersim/ai_001_001")
    """Path to HyperSim folder with extracted scenes."""
    cam_ids: List[str] = ["cam_00"]
    """Camera id/trajectory used"""
    split_factor: float = 0.5
    """Fraction of images to use for training. Remaining for eval."""
    m_per_asset_unit: float = 1e-3
    """Meters per asset unit factor"""
    height: int = 768
    """Image height"""
    width: int = 1024
    """Image width"""

@dataclass
class HyperSim(DataParser):
    """HyperSim DatasetParser"""
    config: HyperSimDataParserConfig
    def _generate_dataparser_outputs(self, split="train"):
        self._load_m_per_asset_unit()
        self._load_scene_metadata()
        self._load_image_ids()
        self._generate_data_splits(split)
        self._create_cam_model()
        self._rescale_scene_with_boundary()

        cameras = Cameras(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                          height=self.config.height, width=self.config.width,
                          camera_to_worlds=self.poses[:, :3, :4],
                          camera_type=CameraType.PERSPECTIVE)
        
        # Scene (x,y,z) centered at origin with -0.5 to 0.5 range after scale + shift
        scene_box = SceneBox(aabb=torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32))

        dataparser_outputs = DataparserOutputs(image_filenames=self.all_image_names, cameras=cameras,
            scene_box=scene_box, dataparser_scale=self.scale_factor, dataparser_transform=self.transform,
            metadata={"depth_filenames": self.all_depth_names,
                      "normal_filenames": self.all_normal_names,
                      "semantic_filenames": self.all_semantic_names,
                      "semantic_instance_filenames": self.all_semantic_instance_names,
                      "entity_id_filenames": self.all_entity_id_names,
                      "m_per_asset_unit": self.config.m_per_asset_unit,
                      "H_orig": self.config.height, "W_orig": self.config.width,
                      "scene_boundary": self.scene_boundary,
                      "xyz_min": self.xyz_min, "xyz_max": self.xyz_max,
                      "M_cam_from_uv": self.M_cam_from_uv})
        return dataparser_outputs

    def _load_m_per_asset_unit(self):
        '''Load conversion factor from scene units to meters'''
        scene_meta = pd.read_csv(self.config.data / '_detail' / 'metadata_scene.csv')
        tmp_idx = (scene_meta['parameter_name'] == 'meters_per_asset_unit')
        self.config.m_per_asset_unit = scene_meta.loc[tmp_idx, 'parameter_value'].iloc[0]

    def _load_scene_metadata(self):
        """Load scene statistics - List of images for each camera trajectory"""
        self.scene_name = self.config.data.parts[-1]
        self.all_metadata_path = self.config.data.parents[0] / "all_scenes_metadata.json"
        
        # If metadata file available, load it
        if self.all_metadata_path.exists():
            print(f'Loading {self.scene_name} metadata from {self.all_metadata_path}...')
            with open(self.all_metadata_path, 'r') as f:
                all_metadata = json.load(f)
            self.scene_metadata = all_metadata[self.scene_name]
        # Else compute metadata for current scene
        else:
            print(f'Extracting {self.scene_name} metadata, as {self.all_metadata_path} not found...')
            self.imgs_root_dir = self.config.data / 'images'
            cams_list = [x.name for x in self.imgs_root_dir.rglob("*") if 'final_hdf5' in x.name].sort()
            # Go over all camera trajectories
            self.scene_metadata = {'cams': {}}
            for cam in cams_list:
                imgs_list = [x.name for x in (self.imgs_root_dir / cam).rglob("*.hdf5")]
                random.shuffle(imgs_list)
                random.shuffle(imgs_list)
                random.shuffle(imgs_list)
                cam_name = '_'.join(cam.split('_')[1:3])
                self.scene_metadata['cams'][cam_name] = {'img_names': imgs_list}

    def _load_image_ids(self):
        """Load images ids for all requested camera trajectories"""
        print(f'Extracting cameras and their corresponding images')

        # By default we use only cam_00 for all scenes, however if not available, switch to cam_01
        if self.cam_ids == ['cam_00'] and 'cam_00' not in self.scene_metadata['cams']:
            self.cam_ids = ['cam_01']
            print('Scene did not have cam_00, switching to use cam_01...')

        # Go over each selected camera trajectory and append its image ids as cam_id.img_id
        self.all_img_ids = {}
        self.cams = []
        for cur_cam_id in self.cam_ids:
            self.cams.append(cur_cam_id)
            self.all_img_ids[cur_cam_id] = []
            for img in self.scene_metadata['cams'][cur_cam_id]['img_names']:
                full_img_name = self.imgs_root_dir / f'scene_{cur_cam_id}_final_hdf5' / img
                if full_img_name.exists() and h5py.is_hdf5(full_img_name):
                    self.all_img_ids[cur_cam_id].append(f'{cur_cam_id}.{img.split(".")[1]}')
            
        # Assert that we have at least one image
        assert len(self.all_img_ids[self.cam_ids[0]]) > 0
        print(f'Loaded {len(self.cam_ids)} cameras')
        print(f'Loaded {sum([len(self.all_img_ids[k]) for k in self.all_img_ids.keys()])} image ids in total')

    def _generate_data_splits(self, split: str = 'train'):
        """ For each camera trajectory, split the images into train/val splits"""
        # Go through each cam and extract ids for specified split\
        self.img_ids = []
        for cam_id in self.all_img_ids.keys():
            cur_img_ids = self.all_img_ids[cam_id]
            split_point = round(self.split_factor * len(cur_img_ids))
            
            # Image ids are already randomized
            if split == 'train':
                cur_img_ids = cur_img_ids[:split_point]
            elif split == 'test' or split == 'val':
                cur_img_ids = cur_img_ids[split_point:]
            elif split == 'all':
                cur_img_ids = cur_img_ids
            else:
                raise ValueError(f"Unknown dataparser split {split}")
            # Sort after split
            cur_img_ids.sort()
            self.img_ids.extend(cur_img_ids)
        self.all_image_names = [self.config.data / 'images' / 'scene_' + x.split('.')[0] +
            '_final_hdf5' / 'frame.' + x.split('.')[-1] + 'color.hdf5' for x in self.img_ids]
        self.all_depth_names = [self.config.data / 'images' / 'scene_' + x.split('.')[0] +
            '_geometry_hdf5' / 'frame.' + x.split('.')[-1] + 'depth_meters.hdf5' for x in self.img_ids]
        self.all_normal_names = [self.config.data / 'images' / 'scene_' + x.split('.')[0] +
            '_geometry_hdf5' / 'frame.' + x.split('.')[-1] + 'normal_bump_world.hdf5' for x in self.img_ids]
        self.all_semantic_names =  [self.config.data / 'images' / 'scene_' + x.split('.')[0] +
            '_geometry_hdf5' / 'frame.' + x.split('.')[-1] + 'semantic.hdf5' for x in self.img_ids]
        self.all_semantic_instance_names = [self.config.data / 'images' / 'scene_' + x.split('.')[0] +
            '_geometry_hdf5' / 'frame.' + x.split('.')[-1] + 'semantic_instance.hdf5' for x in self.img_ids]
        self.all_entity_id_names = [self.config.data / 'images' / 'scene_' + x.split('.')[0] +
            '_geometry_hdf5' / 'frame.' + x.split('.')[-1] + 'render_entity_id.hdf5' for x in self.img_ids]
        print(f'Extracted the {split} which contains {len(self.img_ids)} images')

    def _create_cam_model(self):
        """Load camera intrinsics and extrinsics"""
        print(f'Loading Camera Intrinsics...')
        self.metric_mode = 'asset_units'
        df_camera_parameters_all = pd.read_csv(CAMERA_METADATA_WEBSITE, index_col="scene_name")
        df_ = df_camera_parameters_all.loc[self.scene_name]

        # # Matrix to go from the image space to camera coordinates
        # # Scale to meters by self.M_cam_from_uv *= self.config.m_per_asset_unit
        self.M_cam_from_uv = torch.FloatTensor([
            [df_['M_cam_from_uv_00'], df_['M_cam_from_uv_01'], df_['M_cam_from_uv_02']],
            [df_['M_cam_from_uv_10'], df_['M_cam_from_uv_11'], df_['M_cam_from_uv_12']],
            [df_['M_cam_from_uv_20'], df_['M_cam_from_uv_21'], df_['M_cam_from_uv_22']],
        ])
        # # Matrix to go from the camera coordinates to image space
        # # Scale to meters by self.M_ndc_from_cam /= self.config.m_per_asset_unit
        # self.M_ndc_from_cam = torch.FloatTensor([
        #     [df_['M_proj_00'], df_['M_proj_01'], df_['M_proj_02'], df_['M_proj_03']],
        #     [df_['M_proj_10'], df_['M_proj_11'], df_['M_proj_12'], df_['M_proj_13']],
        #     [df_['M_proj_20'], df_['M_proj_21'], df_['M_proj_22'], df_['M_proj_23']],
        #     [df_['M_proj_30'], df_['M_proj_31'], df_['M_proj_32'], df_['M_proj_33']],
        # ])
        # self.M_uv_from_ndc = torch.FloatTensor(
        #     [[0.5*(self.config.width-1),  0,                          0,   0.5*(self.config.width-1) ],
        #      [0,                         -0.5*(self.config.height-1), 0,   0.5*(self.config.height-1)],
        #      [0,                          0,                          0.5, 0.5                       ],
        #      [0,                          0,                          0,   1.0                       ]])
        
        # Useful info at https://stackoverflow.com/questions/11277501/how-to-recover-view-space-position-given-view-space-depth-value-and-ndc-xy/46118945#46118945
        # FoV from https://github.com/apple/ml-hypersim/blob/main/contrib/mikeroberts3000/python/dataset_generate_camera_parameters_metadata.py#L164
        # https://github.com/apple/ml-hypersim/issues/44 - fx = fy = 886.81
        self.fov_rad_x = df_['settings_camera_fov']
        self.fov_rad_y = 2.0 * np.arctan(self.config.height * np.tan(self.fov_rad_x/2.0) / self.config.width)
        self.fx = self.config.width / 2.0 / math.tan(self.fov_rad_x / 2.0)
        self.fy = self.config.height / 2.0 / math.tan(self.fov_rad_y / 2.0)
        self.cx = (self.config.width - 1.0) / 2.0
        self.cy = (self.config.height - 1.0) / 2.0

        print(f'Loading Camera Extrinsics...')
        # Load (unordered) camera poses for all camera trajectories
        cam_poses_all_cams = {}
        for cam_id in self.cam_ids:
            # Scale to meters - cam_poses[:3, 3] *= self.config.m_per_asset_unit
            cam_poses_all_cams[cam_id] = self._load_cam_poses_single_cam(cam_id)

        # Reorder camera poses to match self.img_ids list
        poses = []
        for img_id in self.img_ids:
            cam_id, frame_id = img_id.split('.')
            frame_id = int(frame_id)
            
            if cam_poses_all_cams[cam_id]['frame_idx'][frame_id] == frame_id:
                orig_img_idx = frame_id
            else:
                orig_img_idx_where = np.where(cam_poses_all_cams[cam_id]['frame_idx'] == frame_id)
                assert len(orig_img_idx_where) == 1
                orig_img_idx = orig_img_idx_where[0].item()
            poses.append(cam_poses_all_cams[cam_id]['poses'][orig_img_idx])
        self.poses = torch.stack(poses)
    
    def _load_cam_poses_single_cam(self, cam_id: str = 'cam_00') -> Dict[str, Any]:
        cam_dir = self.config.data / '_detail' / cam_id
        translation_file = cam_dir / 'camera_keyframe_positions.hdf5'
        rotation_file = cam_dir / 'camera_keyframe_orientations.hdf5'
        frame_idx_file = cam_dir / 'camera_keyframe_frame_indices.hdf5'
        # look_at_file = cam_dir / 'camera_keyframe_look_at_positions.hdf5'
        
        translations = h5py.File(translation_file, 'r')['dataset'][:]
        rotations = h5py.File(rotation_file, 'r')['dataset'][:]
        frame_idx = h5py.File(frame_idx_file, 'r')['dataset'][:]
        # look_at = h5py.File(look_at_file, 'r')['dataset'][:]

        translations = torch.from_numpy(translations).type(torch.FloatTensor)
        rotations = torch.from_numpy(rotations).type(torch.FloatTensor)
        poses = torch.cat((torch.cat((rotations, translations.unsqueeze(-1)), dim=2),
                           torch.zeros((rotations.shape[0], 1, 4))), dim=1)
        poses[:, -1, -1] = 1.0
        cam_poses = {'poses': poses, 'frame_idx': frame_idx}
        return cam_poses
    
    def _rescale_scene_with_boundary(self):
        """Load scene boundary and rescale the scene"""
        # Try to load precomputed if available
        if 'scene_boundary' in self.scene_metadata:
            print(f'Loading scene boudnary from precomputed scene metadata...')
            self.scene_boundary = self.scene_metadata['scene_boundary']
            for k in self.scene_boundary:
                self.scene_boundary[k] = torch.tensor(self.scene_boundary[k])
        else:
            # Otherwise compute it using depth
            # TODO : Port the code to DataManager if you find its not precomputed
            raise NotImplementedError('Scene boundary not precomputed; either code ' \
                                      'its computation from depth or precompute it')

        if 'xyz_cam1p5_min' in self.scene_boundary:
            print(f'Using cropped boundaries...')
            self.xyz_min = self.scene_boundary['xyz_cam1p5_min']
            self.xyz_max = self.scene_boundary['xyz_cam1p5_max']
        else:
            print(f'Using original full boundaries...')
            self.xyz_min = self.scene_boundary['xyz_scene_min']
            self.xyz_max = self.scene_boundary['xyz_scene_max']
        
        # Rescale the pose, because the scene is viewed in [-0.5, 0.5]
        shift = (self.xyz_max + self.xyz_min) / 2
        scale = (self.xyz_max - self.xyz_min).max().item() / 2 * 1.05 # Enlarge a little so content is within        
        self.poses[:, :3, 3] -= shift.unsqueeze(0)
        self.poses[:, :3, 3] /= 2*scale
        
        self.transform = torch.eye(4)
        self.transform[:3, 3] = -shift.unsqueeze(0)
        self.transform = self.transform[:3, :]
        self.scale_factor = 1.0/(2*scale)