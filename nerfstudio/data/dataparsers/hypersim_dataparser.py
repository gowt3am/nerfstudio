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
import math, json, random, h5py, faiss, scipy.linalg
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, List, Dict, Any
from torchtyping import TensorType

import numpy as np
import pandas as pd
import torch
import pytorch3d

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
    cam_ids: List[str] = field(default_factory=lambda: ["cam_00"])
    """Camera id/trajectory used"""
    split_factor: float = 0.5
    """Fraction of images to use for training. Remaining for eval."""
    m_per_asset_unit: float = 1e-3
    """Meters per asset unit factor"""
    height: int = 768
    """Image height"""
    width: int = 1024
    """Image width"""
    few_shot: int = -1
    """Number of images to use for few-shot training, -1 to use all, automatically chooses the right images"""

@dataclass
class HyperSim(DataParser):
    """HyperSim DatasetParser"""
    config: HyperSimDataParserConfig
    def _generate_dataparser_outputs(self, split="train", **kwargs):
        self.test_tuning = kwargs.get("test_tuning", False)
        self.use_pregen_random_views = kwargs.get("pregen_random_views", False)
        self.random_renders_directory = kwargs.get("random_renders_directory", "rendered_test_00")
        self.use_on_the_fly_random_views = kwargs.get("on_the_fly_random_views", False)
        self.rand_pose_type = kwargs.get("rand_pose_type", "closeby")
        self.num_total_random_poses = kwargs.get("num_total_random_poses", 0)
        self.num_random_views_per_batch = kwargs.get("num_random_views_per_batch", 0)
        
        self._load_m_per_asset_unit()
        self._load_scene_metadata()
        self._load_image_ids()
        self._generate_data_splits(split)
        self._create_cam_model()
        self._rescale_scene_with_boundary()

        self.nearest_cam_ids = None
        self.gen_image_names = None
        self.gen_mask_names = None
        self.gen_poses = self.poses.clone()
        if self.test_tuning:
            # Reconstructed images and masks are numbered from 0 to len(val_data), so different naming convention
            img_ids = [x.split('.')[0] + '.{:04d}'.format(y) for (x, y) in
                                    zip(self.img_ids, range(len(self.img_ids)))]
            self.gen_image_names = [str(self.config.data) + '/images/no_filt_rendered_' +
                x.split('.')[0] + '/frame.' + x.split('.')[-1] + '.color.png' for x in img_ids]
            self.gen_mask_names = [str(self.config.data) + '/images/no_filt_rendered_' +
                x.split('.')[0] + '/frame.' + x.split('.')[-1] + '.valid.png' for x in img_ids]
        elif self.use_pregen_random_views:
            self.num_total_random_poses = len(list((self.config.data / 'images'
                                              / self.random_renders_directory).rglob("*.png"))) // 2
            img_ids = ['{:04d}'.format(x) for x in range(self.num_total_random_poses)]
            self.gen_image_names = [None]*len(self.all_image_names) + [
                str(self.config.data) + f'/images/{self.random_renders_directory}/frame.' + x + '.color.png' for x in img_ids]
            self.gen_mask_names = [None]*len(self.all_image_names) + [
                str(self.config.data) + f'/images/{self.random_renders_directory}/frame.' + x + '.valid.png' for x in img_ids]
            self.gen_poses = torch.from_numpy(np.load(self.config.data / 'images'
                                    / f'{self.random_renders_directory}' / 'random_poses.npy')).float()
            self.gen_poses = torch.cat((self.poses, self.gen_poses), dim=0)
            self.poses = self.gen_poses.clone()
        elif self.use_on_the_fly_random_views:
            self.generate_random_poses()
        
        cameras = Cameras(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                          height=self.config.height, width=self.config.width,
                          camera_to_worlds=self.poses[:, :3, :4],
                          camera_type=CameraType.PERSPECTIVE,
                          M_cam_from_uv=self.M_cam_from_uv.unsqueeze(0))
        
        # Scene (x,y,z) centered at origin with -1 to 1 range after scale + shift
        # Typically used are 1.0 or 1.5
        # scene_box = SceneBox(aabb=torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32))
        scene_box = SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32))

        dataparser_outputs = DataparserOutputs(image_filenames=self.all_image_names, cameras=cameras,
            scene_box=scene_box, dataparser_scale=self.scale_factor, dataparser_transform=self.transform,
            metadata={"num_real_images": len(self.all_image_names),
                      "depth_filenames": self.all_depth_names,
                      "normal_filenames": self.all_normal_names,
                      "semantic_filenames": self.all_semantic_names,
                      "semantic_instance_filenames": self.all_semantic_instance_names,
                      "entity_id_filenames": self.all_entity_id_names,
                      "gen_image_filenames": self.gen_image_names,
                      "gen_mask_filenames": self.gen_mask_names,
                      "gen_poses": self.gen_poses,
                      "nearest_cam_ids": self.nearest_cam_ids,
                      "ordered_gen_poses": 1 if self.rand_pose_type == "increasing" else 0,
                      "m_per_asset_unit": self.config.m_per_asset_unit,
                      "H_orig": self.config.height, "W_orig": self.config.width,
                      "scene_boundary": self.scene_boundary,
                      "xyz_min": self.xyz_min, "xyz_max": self.xyz_max,
                      "M_cam_from_uv": self.M_cam_from_uv,
                      "M_ndc_from_cam": self.M_ndc_from_cam,
                      "M_uv_from_ndc": self.M_uv_from_ndc,
                      "M_warp_cam_pts": self.M_warp_cam_pts,
                      "M_p3dcam_from_cam": self.M_p3dcam_from_cam,
                      "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                      "fov_rad_y": self.fov_rad_y, "orig_poses": self.orig_poses})
        return dataparser_outputs

    def _load_m_per_asset_unit(self):
        '''Load conversion factor from scene units to meters'''
        scene_meta = pd.read_csv(self.config.data / '_detail' / 'metadata_scene.csv')
        tmp_idx = (scene_meta['parameter_name'] == 'meters_per_asset_unit')
        self.config.m_per_asset_unit = scene_meta.loc[tmp_idx, 'parameter_value'].iloc[0]
        print(f'Using m_per_asset_unit={self.config.m_per_asset_unit}')

    def _load_scene_metadata(self):
        """Load scene statistics - List of images for each camera trajectory"""
        self.scene_name = self.config.data.parts[-1]
        self.all_metadata_path = self.config.data.parents[0] / "all_scenes_metadata_sparse.json"
        
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
        assert self.config.few_shot in [-1, 8, 15, 30]

        # By default we use only cam_00 for all scenes, however if not available, switch to cam_01
        if self.config.cam_ids == ['cam_00'] and 'cam_00' not in self.scene_metadata['cams']:
            self.config.cam_ids = ['cam_01']
            print('Scene did not have cam_00, switching to use cam_01...')
        
        if self.config.few_shot > 0:
            print(f'Using {self.config.few_shot} images for few-shot training')
            if self.config.few_shot == 8:
                img_set = "few_08"
            elif self.config.few_shot == 15:
                img_set = "few_15"
            elif self.config.few_shot == 30:
                img_set = "few_30"
            else:
                raise ValueError(f"Unknown few-shot value {self.config.few_shot}, choose one of 8, 15, 30")

            self.all_train_ids = {}
            self.all_valid_ids = {}
            for cur_cam_id in self.config.cam_ids:
                self.all_train_ids[cur_cam_id] = []
                for img in self.scene_metadata['cams'][img_set]['train_names']:
                    full_img_name = self.config.data / 'images' / f'scene_{cur_cam_id}_final_hdf5' / img
                    if full_img_name.exists() and h5py.is_hdf5(full_img_name):
                        self.all_train_ids[cur_cam_id].append(f'{cur_cam_id}.{img.split(".")[1]}')
                
                self.all_valid_ids[cur_cam_id] = []
                for img in self.scene_metadata['cams'][img_set]['val_names']:
                    full_img_name = self.config.data / 'images' / f'scene_{cur_cam_id}_final_hdf5' / img
                    if full_img_name.exists() and h5py.is_hdf5(full_img_name):
                        self.all_valid_ids[cur_cam_id].append(f'{cur_cam_id}.{img.split(".")[1]}')

            # Assert that we have at least one image
            assert len(self.all_train_ids[self.config.cam_ids[0]]) > 0
            print(f'Loaded {len(self.config.cam_ids)} cameras')
            print(f'Loaded {sum([len(self.all_train_ids[k]) for k in self.all_train_ids.keys()])} image ids for training')
            print(f'Loaded {sum([len(self.all_valid_ids[k]) for k in self.all_valid_ids.keys()])} image ids for validation')

        else:
            # Go over each selected camera trajectory and append its image ids as cam_id.img_id
            self.all_img_ids = {}
            self.cams = []
            for cur_cam_id in self.config.cam_ids:
                self.cams.append(cur_cam_id)
                self.all_img_ids[cur_cam_id] = []
                for img in self.scene_metadata['cams'][cur_cam_id]['img_names']:
                    full_img_name = self.config.data / 'images' / f'scene_{cur_cam_id}_final_hdf5' / img
                    if full_img_name.exists() and h5py.is_hdf5(full_img_name):
                        self.all_img_ids[cur_cam_id].append(f'{cur_cam_id}.{img.split(".")[1]}')
                
            # Assert that we have at least one image
            assert len(self.all_img_ids[self.config.cam_ids[0]]) > 0
            print(f'Loaded {len(self.config.cam_ids)} cameras')
            print(f'Loaded {sum([len(self.all_img_ids[k]) for k in self.all_img_ids.keys()])} image ids in total')

    def _generate_data_splits(self, split: str = 'train'):
        """ For each camera trajectory, split the images into train/val splits"""
        # Go through each cam and extract ids for specified split
        
        if self.config.few_shot > 0:
            self.img_ids = []
            if split == "train":
                for cam_id in self.all_train_ids.keys():
                    cur_img_ids = self.all_train_ids[cam_id]
                    cur_img_ids.sort()
                    self.img_ids.extend(cur_img_ids)
            elif split == "test" or split == "val":
                for cam_id in self.all_valid_ids.keys():
                    cur_img_ids = self.all_valid_ids[cam_id]
                    cur_img_ids.sort()
                    self.img_ids.extend(cur_img_ids)
            elif split == "all":
                for cam_id in self.all_train_ids.keys():
                    cur_img_ids = self.all_train_ids[cam_id]
                    cur_img_ids.sort()
                    self.img_ids.extend(cur_img_ids)
                for cam_id in self.all_valid_ids.keys():
                    cur_img_ids = self.all_valid_ids[cam_id]
                    cur_img_ids.sort()
                    self.img_ids.extend(cur_img_ids)
            else:
                raise ValueError(f"Unknown dataparser split {split}")
            
        else:
            self.img_ids = []
            for cam_id in self.all_img_ids.keys():
                cur_img_ids = self.all_img_ids[cam_id]
                split_point = round(self.config.split_factor * len(cur_img_ids))
                
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
                
        self.all_image_names = [str(self.config.data) + '/images/scene_' + x.split('.')[0] +
            '_final_hdf5/frame.' + x.split('.')[-1] + '.color.hdf5' for x in self.img_ids]
        self.all_depth_names = [str(self.config.data) + '/images/scene_' + x.split('.')[0] +
            '_geometry_hdf5/frame.' + x.split('.')[-1] + '.depth_meters.hdf5' for x in self.img_ids]
        self.all_normal_names = [str(self.config.data) + '/images/scene_' + x.split('.')[0] +
            '_geometry_hdf5/frame.' + x.split('.')[-1] + '.normal_bump_world.hdf5' for x in self.img_ids]
        self.all_semantic_names =  [str(self.config.data) + '/images/scene_' + x.split('.')[0] +
            '_geometry_hdf5/frame.' + x.split('.')[-1] + '.semantic.hdf5' for x in self.img_ids]
        self.all_semantic_instance_names = [str(self.config.data) + '/images/scene_' + x.split('.')[0] +
            '_geometry_hdf5/frame.' + x.split('.')[-1] + '.semantic_instance.hdf5' for x in self.img_ids]
        self.all_entity_id_names = [str(self.config.data) + '/images/scene_' + x.split('.')[0] +
            '_geometry_hdf5/frame.' + x.split('.')[-1] + '.render_entity_id.hdf5' for x in self.img_ids]
        print(f'Extracted the {split} which contains {len(self.img_ids)} images')
        
    def _create_cam_model(self):
        """Load camera intrinsics and extrinsics"""
        print(f'Loading Camera Intrinsics...')
        self.metric_mode = 'asset_units'
        df_camera_parameters_all = pd.read_csv(CAMERA_METADATA_WEBSITE, index_col="scene_name")
        df_ = df_camera_parameters_all.loc[self.scene_name]
        self.df = df_

        # # Matrix to go from the image space to camera coordinates
        # # Scale to meters by self.M_cam_from_uv *= self.config.m_per_asset_unit
        self.M_cam_from_uv = torch.FloatTensor([
            [df_['M_cam_from_uv_00'], df_['M_cam_from_uv_01'], df_['M_cam_from_uv_02']],
            [df_['M_cam_from_uv_10'], df_['M_cam_from_uv_11'], df_['M_cam_from_uv_12']],
            [df_['M_cam_from_uv_20'], df_['M_cam_from_uv_21'], df_['M_cam_from_uv_22']],
        ])
        # # Matrix to go from the camera coordinates to image space
        # # Scale to meters by self.M_ndc_from_cam /= self.config.m_per_asset_unit
        self.M_ndc_from_cam = torch.FloatTensor([
            [df_['M_proj_00'], df_['M_proj_01'], df_['M_proj_02'], df_['M_proj_03']],
            [df_['M_proj_10'], df_['M_proj_11'], df_['M_proj_12'], df_['M_proj_13']],
            [df_['M_proj_20'], df_['M_proj_21'], df_['M_proj_22'], df_['M_proj_23']],
            [df_['M_proj_30'], df_['M_proj_31'], df_['M_proj_32'], df_['M_proj_33']],
        ])
        self.M_uv_from_ndc = torch.FloatTensor(
            [[0.5*(self.config.width-1),  0,                          0,   0.5*(self.config.width-1) ],
             [0,                         -0.5*(self.config.height-1), 0,   0.5*(self.config.height-1)],
             [0,                          0,                          0.5, 0.5                       ],
             [0,                          0,                          0,   1.0                       ]])
        
        # Useful info at https://stackoverflow.com/questions/11277501/how-to-recover-view-space-position-given-view-space-depth-value-and-ndc-xy/46118945#46118945
        # FoV from https://github.com/apple/ml-hypersim/blob/main/contrib/mikeroberts3000/python/dataset_generate_camera_parameters_metadata.py#L164
        # https://github.com/apple/ml-hypersim/issues/44 - fx = fy = 886.81
        if df_["use_camera_physical"]:
            print("Using Camera Physical FOV")
            self.fov_rad_x = df_["camera_physical_fov"]
        else:
            print("Using Camera Settings FOV")
            self.fov_rad_x = df_["settings_camera_fov"]
        self.fov_rad_y = 2.0 * np.arctan(self.config.height * np.tan(self.fov_rad_x/2.0) / self.config.width)
        self.fx = self.config.width / 2.0 / math.tan(self.fov_rad_x / 2.0)
        self.fy = self.config.height / 2.0 / math.tan(self.fov_rad_y / 2.0)
        self.cx = (self.config.width - 1.0) / 2.0
        self.cy = (self.config.height - 1.0) / 2.0
        self.K = torch.tensor([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        # DO NOT USE THIS K (only works for some scenes!!), use M_cam_from_uv instead

        M_cam_from_uv_canonical = torch.tensor([[np.tan(self.fov_rad_x/2.0), 0.0, 0.0],
                                                [0.0, np.tan(self.fov_rad_y/2.0), 0.0],
                                                [0.0, 0.0, -1.0]], dtype=torch.float32)

        # [ONLY FOR PyTorch3D] has problems with non-standard perspective projection matrices from Hypersim
        # so we construct a matrix to transform a camera-space points to warped locations,
        # such that the warped position can be projected with a standard perspective projection matrix
        M_warp_cam_pts_ = M_cam_from_uv_canonical @ torch.linalg.inv(self.M_cam_from_uv)
        self.M_warp_cam_pts  = torch.from_numpy(scipy.linalg.block_diag(M_warp_cam_pts_, 1)).float()
        # Transform from Right-Up-Back (Cam to World) format of Hypersim to
        # Left-Up-Front (World to Cam) format of PyTorch3D
        self.M_p3dcam_from_cam = torch.eye(4)
        self.M_p3dcam_from_cam[0,0] = -1
        self.M_p3dcam_from_cam[2,2] = -1

        print(f'Loading Camera Extrinsics...')
        # Load (unordered) camera poses for all camera trajectories
        cam_poses_all_cams = {}
        for cam_id in self.config.cam_ids:
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
            print(f'Loading scene boundary from precomputed scene metadata...')
            self.scene_boundary = self.scene_metadata['scene_boundary']
            for k in self.scene_boundary:
                self.scene_boundary[k] = torch.tensor(self.scene_boundary[k])
        else:
            # Otherwise compute it using depth
            # TODO : Port the code to DataManager if you find its not precomputed
            raise NotImplementedError('Scene boundary not precomputed; either code ' \
                                      'its computation from depth or precompute it')

        if 'xyz_manual_min' in self.scene_boundary:
            print(f'Using Gowtham manually cropped boundaries...')
            self.xyz_min = self.scene_boundary['xyz_manual_min']
            self.xyz_max = self.scene_boundary['xyz_manual_max']
        elif 'xyz_cam1p5_min' in self.scene_boundary:
            print(f'Using Nikola\'s algorithm cropped boundaries...')
            self.xyz_min = self.scene_boundary['xyz_cam1p5_min']
            self.xyz_max = self.scene_boundary['xyz_cam1p5_max']
        else:
            print(f'Using original full boundaries...')
            self.xyz_min = self.scene_boundary['xyz_scene_min']
            self.xyz_max = self.scene_boundary['xyz_scene_max']
        
        # Rescale the pose, because the scene is viewed in [-1, 1]
        shift = (self.xyz_max + self.xyz_min) / 2
        scale = (self.xyz_max - self.xyz_min).max().item() / 2.0 * 1.2 # Enlarge a little, so main scene is fully enclosed
        scale = scale / 2.0   # 2.0 scales the scene to [-1, 1]. If 4.0: [-2, 2]
        self.orig_poses = self.poses.clone()
        self.poses[:, :3, 3] -= shift.unsqueeze(0)
        self.poses[:, :3, 3] /= 2*scale
        
        self.transform = torch.eye(4)
        self.transform[:3, 3] = -shift.unsqueeze(0)
        self.transform = self.transform[:3, :]
        self.scale_factor = 1.0/(2*scale)

    def generate_random_poses(self):
        """Generate all random poses that will be used for training"""
        
        if self.rand_pose_type == "closeby":
            self.gen_poses, self.nearest_cam_ids = self._closeby_poses(
                                    self.poses, num_poses=self.num_total_random_poses)
        elif self.rand_pose_type == "sparf":
            self.gen_poses, self.nearest_cam_ids = self._sparf_poses(
                                    self.poses, num_poses=self.num_total_random_poses)
        elif self.rand_pose_type == "increasing":
            self.gen_poses, self.nearest_cam_ids = self._slowly_increasing_poses(
                self.poses, num_poses=self.num_total_random_poses, num_trajectories=self.num_random_views_per_batch)
        else:
            raise ValueError(f"Unknown rand_pose_type {self.rand_pose_type}")
        self.nearest_cam_ids = np.concatenate((np.arange(self.poses.shape[0]), self.nearest_cam_ids))
        self.gen_poses = torch.cat((self.poses, self.gen_poses), dim=0).cuda()
        self.poses = self.gen_poses.clone()
    
    def _normalize(self, x):
            """Normalization helper function."""
            return x / np.linalg.norm(x)

    def _viewmatrix(self, lookdir, up, position, subtract_position=False):
        """Construct lookat view matrix."""
        vec2 = self._normalize((lookdir - position) if subtract_position else lookdir)
        vec0 = self._normalize(np.cross(up, vec2))
        vec1 = self._normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def _closeby_poses(self, poses: TensorType, num_poses: int = 1500):
        print(f'Generating {num_poses} random poses from trainset poses, closest to training views...')

        # Poses are in Right-Up-Back (Cam to World) format
        all_poses = poses.numpy()
        all_ups = all_poses[:, :3, 1]
        all_origins = all_poses[:, :3, 3]
        all_z_axes = - all_poses[:, :3, 2]
        all_lookat = all_origins + all_z_axes

        # Using faiss to cluster all_origins into 2 clusters, and sample pairs
        # separately from each cluster - Helps for some scenes
        kmeans = faiss.Kmeans(d=3, k=2, niter=20, verbose=False)
        kmeans.train(all_origins.astype(np.float32))
        _, I = kmeans.index.search(all_origins.astype(np.float32), 1)
        cluster1 = {i: all_origins[i] for i in range(all_origins.shape[0]) if I[i] == 0}
        cluster2 = {i: all_origins[i] for i in range(all_origins.shape[0]) if I[i] == 1}

        # Randomly select 2 different camera ups, and lineraly interpolates between them
        idx1 = np.random.choice(all_ups.shape[0], size=num_poses*2, replace=True)
        ratios = np.random.rand(num_poses)
        up1 = all_ups[idx1[:num_poses]]
        up2 = all_ups[idx1[num_poses:]]
        new_ups = ratios[:, None] * up1 + (1 - ratios[:, None]) * up2

        """
        Randomly selects 2 different camera origins from the same cluster (to ensure not
        in different parts of room), and lineraly interpolates both origin and z-axis
        between them.  
        
        Goal is to generate camera poses that are close to the original views.
        """
        idx1 = np.random.choice(all_origins.shape[0], size=num_poses, replace=True)
        idx2 = np.array([np.random.choice(list(cluster1.keys())) if I[i] == 0 
                            else np.random.choice(list(cluster2.keys())) for i in idx1])
        ratios = np.random.rand(num_poses)
        
        origin1 = all_origins[idx1]
        origin2 = all_origins[idx2]
        new_origins = ratios[:, None] * origin1 + (1 - ratios[:, None]) * origin2
        
        new_lookats = np.where(ratios[:, None] < 0.5, all_lookat[idx2], all_lookat[idx1])
        new_z_axes = new_lookats - new_origins
        render_camera_idxs = np.where(ratios < 0.5, idx2, idx1)
        
        new_poses = []
        for i in range(num_poses):
            new_poses.append(self._viewmatrix(new_z_axes[i], new_ups[i], new_origins[i]))
        
        # Poses are in Left-Up-Front (Cam to World) format, convert it to Right-Up-Back (Cam to World) format
        new_poses = np.stack(new_poses, axis=0)
        new_poses[:, :, 0] = - new_poses[:, :, 0]
        new_poses[:, :, 2] = - new_poses[:, :, 2]
        new_poses = np.concatenate([new_poses, np.zeros((num_poses, 1, 4))], axis=1)
        new_poses[:, 3, 3] = 1.0
        return torch.from_numpy(new_poses).float(), render_camera_idxs
    
    def _sparf_poses(self, poses: TensorType, num_poses: int = 1500):
        print(f'Generating {num_poses} random poses from trainset poses, using SPARF sampling...')

        # Poses are in Right-Up-Back (Cam to World) format
        all_poses = poses.numpy()
        all_ups = all_poses[:, :3, 1]
        all_origins = all_poses[:, :3, 3]
        all_z_axes = - all_poses[:, :3, 2]
        all_lookat = all_origins + all_z_axes

        """
        Randomly select a camera pose, find the closest camera and linearly interpolate 
        between them. Use the lookat direction of whichever camera factor is more.
        
        Goal is to generate camera poses that are as good as the original views
        """
        idx1 = np.random.choice(all_origins.shape[0], size=num_poses, replace=True)
        chosen_origins1 = all_origins[idx1].copy()
        chosen_lookat1 = all_lookat[idx1].copy()
        chosen_ups1 = all_ups[idx1].copy()

        idx2 = []
        for i in range(num_poses):
            dists = np.linalg.norm(all_origins - chosen_origins1[i], axis=1)
            dists[idx1[i]] = np.inf
            idx2.append(np.argmin(dists))
        choosen_origins2 = all_origins[idx2].copy()
        choosen_lookat2 = all_lookat[idx2].copy()
        choosen_ups2 = all_ups[idx2].copy()

        ratios = np.random.rand(num_poses)
        new_origins = ratios[:, None] * chosen_origins1 + (1 - ratios[:, None]) * choosen_origins2
        new_ups = ratios[:, None] * chosen_ups1 + (1 - ratios[:, None]) * choosen_ups2
        new_lookats = np.where(ratios[:, None] < 0.5, choosen_lookat2, chosen_lookat1)
        new_z_axes = new_lookats - new_origins
        render_camera_idxs = np.where(ratios < 0.5, idx2, idx1)

        new_poses = []
        for i in range(num_poses):
            new_poses.append(self._viewmatrix(new_z_axes[i], new_ups[i], new_origins[i]))
        
        # Poses are in Left-Up-Front (Cam to World) format, convert it to Right-Up-Back (Cam to World) format
        new_poses = np.stack(new_poses, axis=0)
        new_poses[:, :, 0] = - new_poses[:, :, 0]
        new_poses[:, :, 2] = - new_poses[:, :, 2]
        new_poses = np.concatenate([new_poses, np.zeros((num_poses, 1, 4))], axis=1)
        new_poses[:, 3, 3] = 1.0
        return torch.from_numpy(new_poses).float(), render_camera_idxs
    
    def _slowly_increasing_poses(self, poses: TensorType, num_poses: int = 1500, num_trajectories: int = 50):
        print(f'Generating {num_poses} random poses that slowly move away from trainset poses...')

        # Poses are in Right-Up-Back (Cam to World) format
        all_poses = poses.numpy()
        all_ups = all_poses[:, :3, 1]
        all_origins = all_poses[:, :3, 3]
        all_z_axes = - all_poses[:, :3, 2]
        all_lookat = all_origins + all_z_axes
    
        """
        Randomly select num_trajectories farthest camera poses, keep them as the starting point.
        Find distance to nearest camera (this is a proxy for distance we can move freely).
        Now randomly sample a 3D direction, and sample num_per_trajectory points along the line,
        still looking at the original lookat direction.
        
        Goal is to generate increasingly complicated views so network learns well.
        """
        chosen_origins1, idx1 = pytorch3d.ops.sample_farthest_points(
                                points=torch.from_numpy(all_origins).unsqueeze(0),
                                lengths=None, K=num_trajectories)
        chosen_origins1 = chosen_origins1.squeeze(0).numpy()
        idx1 = idx1.squeeze(0).numpy()
        chosen_lookat1 = all_lookat[idx1].copy()
        chosen_ups1 = all_ups[idx1].copy()

        idx2 = []
        for i in range(chosen_origins1.shape[0]):
            dists = np.linalg.norm(all_origins - chosen_origins1[i], axis=1)
            dists[idx1[i]] = np.inf
            idx2.append(np.argmin(dists))
        choosen_origins2 = all_origins[idx2].copy()
        dists = np.linalg.norm(chosen_origins1 - choosen_origins2, axis=1)
        
        random_directions = np.random.randn(num_trajectories, 3)
        random_directions = random_directions / np.linalg.norm(random_directions, axis=1)[:, None]
        random_directions = random_directions * dists[:, None]
        end_points = chosen_origins1 + random_directions
        
        new_poses = np.zeros((num_poses, 3, 4))
        render_camera_idxs = np.zeros((num_poses), dtype=np.int32)
        num_views_per_trajectory = num_poses // num_trajectories
        for i in range(num_trajectories):
            # Sample num_per_trajectory points along the line between chosen_origins1 and end_points
            ratios = np.linspace(0, 1, num_views_per_trajectory)
            
            new_origins = (1 - ratios[:, None]) * chosen_origins1[i] + ratios[:, None] * end_points[i]
            new_ups = np.repeat(chosen_ups1[i][None, :], num_views_per_trajectory, axis=0)
            new_lookats = np.repeat(chosen_lookat1[i][None, :], num_views_per_trajectory, axis=0)
            new_z_axes = new_lookats - new_origins
            
            for j in range(num_views_per_trajectory):
                new_poses[j*num_trajectories + i] = self._viewmatrix(new_z_axes[j], new_ups[j], new_origins[j])
                render_camera_idxs[j*num_trajectories + i] = idx1[i]

        # Poses are in Left-Up-Front (Cam to World) format, convert it to Right-Up-Back (Cam to World) format
        new_poses = np.stack(new_poses, axis=0)
        new_poses[:, :, 0] = - new_poses[:, :, 0]
        new_poses[:, :, 2] = - new_poses[:, :, 2]
        new_poses = np.concatenate([new_poses, np.zeros((num_poses, 1, 4))], axis=1)
        new_poses[:, 3, 3] = 1.0
        return torch.from_numpy(new_poses).float(), render_camera_idxs