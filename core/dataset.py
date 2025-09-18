# main.py
# opengl/blender -> colmap style
# use opengl for Plucker Embedding
# OpenGL (x=Right, y=Up, z=Backward (camera looks along −Z))
# Colmap (x=Right, y=Down, z=Forward (camera looks along +Z))


import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from scipy.stats import gamma
from typing import Tuple, Literal, Dict, Optional



from kiui.cam import orbit_camera
from core.model_config import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ObjaverseDataset(Dataset):
    def __init__(self, data_path, cfg: Options, type: Literal['train', 'test', 'val']='train'):
        
        self.data_path = data_path
        self.cfg = cfg
        self.type = type if type in ['train', 'test', 'val'] else 'train'

        
        self.subfolder = [os.path.join(data_path, sub) for sub in os.listdir(data_path) 
                          if os.path.isdir(os.path.join(data_path, sub))]
        
        self.items = []
        self.unused_items = [
            '/kaggle/input/objaverse-subset/archive_60/d7380c30ab55424e9f770c9115c56a83',
            '/kaggle/input/objaverse-subset/archive_17/279673bdc0c549df99a51f8469bae811',
            '/kaggle/input/objaverse-subset/archive_62/ff70344b758b44ddb3c4e31b7eab91be',
            '/kaggle/input/objaverse-subset/archive_27/b6d50c45ddca470b938a48cf80612470',
            '/kaggle/input/objaverse-subset/archive_65/b7b851ecb844419ca4c0b780a62b27ae',
            '/kaggle/input/objaverse-subset/archive_53/caa7053c0ee64ce8ac7ed1c8276af0de',
            '/kaggle/input/objaverse-subset/archive_55/f57e883babaa4369aa0ecf09bbea04b0',
            '/kaggle/input/objaverse-subset/archive_100/f5836f1fadd54919be5ce641182fd386',
            '/kaggle/input/objaverse-subset/archive_96/2d2c2f9f1c54400187eea823843f8e2e',
            '/kaggle/input/objaverse-subset/archive_57/65791b708f5d45dab22fb6be46255854',
            '/kaggle/input/objaverse-subset/archive_57/d4f7e28fabb448049530a86c8dae568b',
            '/kaggle/input/objaverse-subset/archive_5/793f1bd80bfb45268832b267d6a31cab',
            '/kaggle/input/objaverse-subset/archive_5/b637b4e3f43d414c96edb6a0c18b0603'
        ]

        for sub in self.subfolder:
            for item in os.listdir(sub):
                item_path = os.path.join(sub, item)
                if os.path.isdir(item_path) and item_path not in self.unused_items:
                    self.items.append(item_path)

        # naive split
        if self.type == 'val':
            self.items = self.items[-int(self.cfg.val_size * len(self.items)):]
        elif self.type == 'test':
            self.items = self.items[-int((self.cfg.val_size + self.cfg.test_size) * len(self.items)):-int(self.cfg.val_size * len(self.items) - 1)]
        else:
            self.items = self.items[:int(self.cfg.train_size * len(self.items))]

        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.cfg.fovy))
        self.projection_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.projection_matrix[0, 0] = 1 / self.tan_half_fov
        self.projection_matrix[1, 1] = 1 / self.tan_half_fov
        self.projection_matrix[2, 2] = (self.cfg.zfar + self.cfg.znear) / (self.cfg.zfar - self.cfg.znear)
        self.projection_matrix[3, 2] = - (self.cfg.zfar * self.cfg.znear) / (self.cfg.zfar - self.cfg.znear)
        self.projection_matrix[2, 3] = 1

        self.input_view_ids = [0, 2, 4, 6,          # L1
                               9, 11, 13, 15,       # L2
                               16, 18, 20, 22,      # L3    
                               24]                  # L4
        
        self.test_view_ids = [i for i in range(cfg.num_views_total)]
        self.cam_config = {
            # this is the params to pass into kiui.orbit_camera() function
            # (elevation, azimuth)
            # elevation = 0
            0: [0, 0],
            1: [0, 45],
            2: [0, 90],
            3: [0, 135],
            4: [0, 180],
            5: [0, 225],
            6: [0, 270],
            7: [0, 315],

            # elevation = 30
            8: [30, 0],
            9: [30, 45],
            10: [30, 90],
            11: [30, 135],
            12: [30, 180],
            13: [30, 225],
            14: [30, 270],
            15: [30, 315],

            # elevation = 60
            16: [60, 0],
            17: [60, 45],
            18: [60, 90],
            19: [60, 135],
            20: [60, 180],
            21: [60, 225],
            22: [60, 270],
            23: [60, 315],

            # elevation = 90,
            24: [89.89, 180]
        }


    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        #  NEED TO PROCESS DATA IN .OBJ FORMAT TO (IMAGE-CAMERA POSE) PAIRS
        # your_dataset/
            # ├── uid/
            # │   ├── rgb/
            # │   │   ├── 000.png
            # │   │   ├── 001.png
            # │   ├── pose/
            # │   │   ├── 000.txt
            # │   │   ├── 001.txt

        assert len(self.input_view_ids) == self.cfg.num_views_input

        item_path = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        view_ids = self.input_view_ids + np.random.permutation(self.test_view_ids).tolist()
        view_ids = view_ids[:(self.cfg.num_views_input + self.cfg.num_views_output)]

        def find_nonzero_bbox(alpha_channel):
            """Find bounding box (ymin, ymax, xmin, xmax) where alpha > 0."""
            ys, xs = np.where(alpha_channel > 0.000001)
            if len(xs) == 0 or len(ys) == 0:  # Fully transparent
                return None
            return ys.min(), ys.max(), xs.min(), xs.max()

        global_ymin, global_ymax = 1e9, -1
        global_xmin, global_xmax = 1e9, -1
        for view_id in view_ids:
        
            # data path: /kaggle/input/objaverse-subset/archive_4
            image_path = os.path.join(item_path, 'rgb', f'{view_id:03d}.png')
            camera_path = os.path.join(item_path, 'pose', f'{view_id:03d}.txt') 

            # try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # shape: [512, 512, 4]
            alpha = image[:, :, 3]
            bbox = find_nonzero_bbox(alpha)
            if bbox is None:
                print(f"Fully transparent image at {item_path}")
                bbox = (1e9, -1, 1e9, -1)
            
            ymin, ymax, xmin, xmax = bbox
            global_ymin = min(global_ymin, ymin)
            global_ymax = max(global_ymax, ymax)
            global_xmin = min(global_xmin, xmin)
            global_xmax = max(global_xmax, xmax)

            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image)  # shape: [H, W, C]
            
            c2w = torch.from_numpy(orbit_camera(-self.cam_config[view_id][0], self.cam_config[view_id][1], radius=self.cfg.cam_radius, opengl=True))

            # scale up radius to make model make scale predictions
            c2w[:3, 3] *= self.cfg.cam_radius / 1.5 # 1.5 is the default scale of the dataset
        
            # Background removing
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

        origin_size = images[0].shape[1]
        res_ymax = origin_size - global_ymax
        res_ymin = global_ymin
        res_xmax = origin_size - global_xmax
        res_xmin = global_xmin
        min_res = min(res_ymax, min(res_ymin, min(res_xmax, res_xmin)))
        images = [image[:, min_res:(origin_size - min_res), min_res:(origin_size - min_res)]
                  for image in images]
        masks = [mask[min_res:(origin_size - min_res), min_res:(origin_size - min_res)]
                  for mask in masks]

        view_cnt = len(images)
        if view_cnt < (self.cfg.num_views_input + self.cfg.num_views_output):
            print(f'[WARN] dataset {item_path}: not enough valid views, only {view_cnt} views found!')
            # Padding to be enough views
            n = (self.cfg.num_views_input + self.cfg.num_views_output) - view_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n

        images = torch.stack(images, dim=0)     # [V, C, H, W]
        masks = torch.stack(masks, dim=0)       # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0)  # [V, 4, 4]

        # # normalized camera feats as in paper (transform the first pose to a fixed position)
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.cfg.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        # cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        # resize input images
        images_input = F.interpolate(images[:len(self.input_view_ids)].clone(), size=(self.cfg.input_size, self.cfg.input_size), mode='bilinear', align_corners=False)   # [V, C, H, W]
        cam_poses_input = cam_poses[:len(self.input_view_ids)].clone()
        
        # # data augmentation
        # if self.type == 'train':
        #     # if random.random() < self.cfg.prob_grid_distortion:
        #     #     images_input[1:] = grid_distortion(images_input[1:])
        #     if random.random() < self.cfg.prob_cam_jitter:
        #         cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # build rays for input views
        rays_embeddings = []
        for i in range(len(self.input_view_ids)):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.cfg.input_size, self.cfg.input_size, self.cfg.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V=9, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=9, 9, H, W]

        results['input'] = final_input
        results['cam_poses_input'] = cam_poses_input

        # resize ground-truth images, still in range [0, 1]
        results['images_output'] = F.interpolate(images[len(self.input_view_ids):].clone(), (self.cfg.output_size, self.cfg.output_size), mode='bilinear', align_corners=False)
        results['masks_output'] = F.interpolate(masks[len(self.input_view_ids):].clone().unsqueeze(1), (self.cfg.output_size, self.cfg.output_size), mode='bilinear', align_corners=False)

        cam_poses = cam_poses[len(self.input_view_ids):].clone()
        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2)     # World-to-camera matrix: [V, 4, 4] (row-vector)
        cam_view_proj = cam_view @ self.projection_matrix     # world-to-clip matrix: [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view_output'] = cam_view
        results['cam_view_proj_output'] = cam_view_proj
        results['cam_pos_output'] = cam_pos

        # results = {
        #     [C, H, W]
        #     'input': ...,             (processed input images 5x9x256x256)
        #     'cam_poses_input': ...,   
        #     'images_output': ...,     (9x3x512x512)
        #     'masks_output': ...,      (.......)
        #     'cam_view_output': ...,          (colmap coordinate)
        #     'cam_view_proj_output': ...,     (colmap coordinate)
        #     'cam_pos_output': ...,           (colmap coordinate)
        # }
        return results