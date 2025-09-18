# core/model.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import roma
from kiui.op import safe_normalize

def get_rays(pose, h, w, fovy, opengl=True):
    # pose (4, 4): camera-to-world matrix coordinate
    # with pose[:3, :3] determine the direction of camera, pose[:3, 3]  determine the position of camera.
    # h, w: image height and width
    # fovy: vertical field of view in degrees
    # opengl: whether to use OpenGL convention (Y axis flipped)

    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                # currently x is range from 0 to w
                # we move the coordinate system from top-left to the center
                # but the coodinate of each pixel is in the top-left of them, we +0.5 to make sure the coordinate is in the center of each pixel
                # and /focal, all you need to know it that it's used to standardize to the virtual image, which have the distance of 1 to the center of camera
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    # convert camera direction to world space
    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
    # get the translation of camera origin related to world space to get origin ray position in world space
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) # [1, 3] -> [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    # (origin of each ray, direction of each ray)
    return rays_o, rays_d

def orbit_camera_jitter(poses, strength=0.1):
    # pose (B, 4, 4): camera-to-world matrix coordinate, assume orbit camera in opengl format
    # random orbital rotate

    B = poses.shape[0]
    # rotate x around y, and y around x, and the max rotate of y is only half of x
    rotvec_x = poses[:, :3, 1] * strength * np.pi * (torch.rand(B, 1, device=poses.device) * 2 - 1)
    rotvec_y = poses[:, :3, 0] * strength * np.pi / 2 * (torch.rand(B, 1, device=poses.device) * 2 - 1)

    rot = roma.rotvec_to_rotmat(rotvec_x) @ roma.rotvec_to_rotmat(rotvec_y)
    # when direction change, position must change at the same time
    R = rot @ poses[:, :3, :3]
    T = rot @ poses[:, :3, 3:]

    new_poses = poses.clone()
    new_poses[:, :3, :3] = R
    new_poses[:, :3, 3:] = T
    
    return new_poses

def grid_distortion(images, strength=0.5):
    # images: [B, C, H, W]
    # num_steps: int, grid resolution for distortion
    # strength: float in [0, 1], strength of distortion

    # USED FOR IMAGE PIXEL DISTORTION

    B, C, H, W = images.shape

    num_steps = np.random.randint(8, 17)
    grid_steps = torch.linspace(-1, 1, num_steps)

    # have to loop batch...
    grids = []
    for b in range(B):
        # construct displacement
        x_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
        x_steps = (x_steps + strength * (torch.rand_like(x_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
        x_steps = (x_steps * W).long() # [num_steps]
        x_steps[0] = 0
        x_steps[-1] = W
        xs = []
        for i in range(num_steps - 1):
            xs.append(torch.linspace(grid_steps[i], grid_steps[i + 1], x_steps[i + 1] - x_steps[i]))
        xs = torch.cat(xs, dim=0) # [W]

        y_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
        y_steps = (y_steps + strength * (torch.rand_like(y_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
        y_steps = (y_steps * H).long() # [num_steps]
        y_steps[0] = 0
        y_steps[-1] = H
        ys = []
        for i in range(num_steps - 1):
            ys.append(torch.linspace(grid_steps[i], grid_steps[i + 1], y_steps[i + 1] - y_steps[i]))
        ys = torch.cat(ys, dim=0) # [H]

        # construct grid
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy') # [H, W]
        grid = torch.stack([grid_x, grid_y], dim=-1) # [H, W, 2]

        grids.append(grid)
    
    grids = torch.stack(grids, dim=0).to(images.device) # [B, H, W, 2]

    # grid sample
    images = F.grid_sample(images, grids, align_corners=False)

    return images


