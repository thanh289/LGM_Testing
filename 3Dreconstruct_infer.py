from core.model_config import AllConfigs, Options
from core.model import LGM
from accelerate import Accelerator
from safetensors.torch import load_file
from core.dataset import ObjaverseDataset as Dataset
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR
from kiui.cam import orbit_camera
from core.utils import get_rays, grid_distortion, orbit_camera_jitter


import torch
import tyro
import kiui
import wandb
import numpy as np
import os
import random
import imageio
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as TF

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

cfg = tyro.cli(AllConfigs)

model = LGM(cfg)

# Load model checkpoint for FINE-TUNING
if cfg.fine_tune and cfg.resume is not None:
    # (cfg.resume in file type)
    if cfg.resume.endswith('safetensors'):
        ckpt = load_file(cfg.resume, device='cpu')
    else:
        ckpt = torch.load(cfg.resume, map_location='cpu')
    
    # tolerant load (only load matching shapes)
    # model.load_state_dict(ckpt, strict=False)
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
        else:
            print(f'[WARN] unexpected param {k}: {v.shape}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

tan_half_fov = np.tan(0.5 * np.deg2rad(cfg.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (cfg.zfar + cfg.znear) / (cfg.zfar - cfg.znear)
proj_matrix[3, 2] = - (cfg.zfar * cfg.znear) / (cfg.zfar - cfg.znear)
proj_matrix[2, 3] = 1

cam_input_config = {
    0: [0, 0],
    2: [0, 90],
    4: [0, 180],
    6: [0, 270],
    24: [89.89, 180]
}

view_ids = [0, 2, 4, 6, 24]

def find_nonzero_bbox(alpha_channel):
    """Find bounding box (ymin, ymax, xmin, xmax) where alpha > 0."""
    ys, xs = np.where(alpha_channel > 0.000001)
    if len(xs) == 0 or len(ys) == 0:  # Fully transparent
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()

def run(cfg: Options, path):
    images = []
    masks = []
    cam_poses = []
    os.makedirs(cfg.workspace, exist_ok=True)

    # ====================== Data loading and preprocessing ======================
    global_ymin, global_ymax = 1e9, -1
    global_xmin, global_xmax = 1e9, -1
    for view_id in view_ids:
        image_path = os.path.join(path, 'rgb', f'{view_id:03d}.png')
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # shape: [512, 512, 4]
        alpha = image[:, :, 3]
        
        bbox = find_nonzero_bbox(alpha)
        if bbox is None:
            print(f"Fully transparent image at {path}")
            bbox = (1e9, -1, 1e9, -1)
            
        ymin, ymax, xmin, xmax = bbox
        global_ymin = min(global_ymin, ymin)
        global_ymax = max(global_ymax, ymax)
        global_xmin = min(global_xmin, xmin)
        global_xmax = max(global_xmax, xmax)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)  # shape: [H, W, C]
        
        c2w = torch.from_numpy(orbit_camera(-cam_input_config[view_id][0], cam_input_config[view_id][1], radius=cfg.cam_radius, opengl=True))
        c2w[:3, 3] *= cfg.cam_radius / 1.5 # 1.5 is the default scale of the dataset

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
    
    images = torch.stack(images, dim=0)     # [V, C, H, W]
    masks = torch.stack(masks, dim=0)       # [V, H, W]
    cam_poses = torch.stack(cam_poses, dim=0)  # [V, 4, 4]
    images = F.interpolate(images, size=(cfg.input_size, cfg.input_size), mode='bilinear', align_corners=False)
    images = TF.normalize(images, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    rays_embeddings = []
    for i in range(len(view_ids)):
        rays_o, rays_d = get_rays(cam_poses[i], cfg.input_size, cfg.input_size, cfg.fovy) # [h, w, 3]
        rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
        rays_embeddings.append(rays_plucker)

    rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V=9, 6, h, w]
    input_images = torch.cat([images, rays_embeddings], dim=1) # [V, 9, H, W]

    # =========================== Inference ===========================
    with torch.no_grad():
        input_images = input_images.unsqueeze(0).to(device) # [1, V, 9, H, W]
        gaussians = model.forward_gaussians(input_images)

        images = []
        elevation = 30
        azimuth = np.arange(0, 720, 4, dtype=np.int32)

        for azi in tqdm(azimuth):

            # c2w matrix
            cam_poses = torch.from_numpy(orbit_camera(-elevation, azi, radius=cfg.cam_radius, opengl=True)).unsqueeze(0).to(device)

            # from OpenGL to COLMAP c2w matrix (COLMAP cam -> OpenGL world)
            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4] --- w2c matrix (OpenGL world -> COLMAP cam)
            cam_view_proj = cam_view @ proj_matrix # [V, 4, 4] --- w2c2clip matrix  (OpenGL world -> COLMAP cam -> image)
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            if cfg.fancy_video:
                scale = min(azi / 720, 1)
            else:
                scale = 1

            image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
            images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(os.path.join(cfg.workspace, path.split('/')[-1] + '.mp4'), images, fps=30)
        

path = '/kaggle/input/10k-objaverse-object/archive_10/003d816301084f719897503bdb4ffbc3'
run(cfg, path)