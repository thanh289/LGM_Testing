# main.py

import kiui.vis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.model_config import Options
from core.unet import UNet
from core.gs import GaussianRenderer
from kiui.lpips import LPIPS
from core.utils import get_rays
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure



class LGM(nn.Module):
    def __init__(self, cfg: Options):
        super().__init__()

        self.cfg = cfg

        # UNet
        self.unet = UNet(
            9, 14, 
            down_channels=self.cfg.down_channels,
            down_attention=self.cfg.down_attention,
            mid_attention=self.cfg.mid_attention,
            up_channels=self.cfg.up_channels,
            up_attention=self.cfg.up_attention,
        )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1)

        # Gaussian Renderer
        self.gs = GaussianRenderer(cfg)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)     # Dense Gaussians
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        # self.opacity_act = lambda x: torch.ones_like(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x) # NOTE: may use sigmoid if train again

        self.lpips_loss = LPIPS(net='vgg')
        self.lpips_loss.requires_grad_(False)

        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # ignore lpips_loss mismatch
        missing, unexpected = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if missing:
            print(f"[Warning] Ignored missing keys: {missing}")
        if unexpected:
            print(f"[Warning] Ignored unexpected keys: {unexpected}")
        return missing, unexpected

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict
    
    def prepare_default_rays(self, device, elevation=0):
        # prepare Plucker embedding for 4 input images

        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.cfg.cam_radius),
            orbit_camera(elevation, 90, radius=self.cfg.cam_radius),
            orbit_camera(elevation, 180, radius=self.cfg.cam_radius),
            orbit_camera(elevation, 270, radius=self.cfg.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.cfg.input_size, self.cfg.input_size, self.cfg.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
    
    def forward_gaussians(self, images):
        # images: [B, 5, 9, H, W]
        # return: Gaussians: [B, num_gauss * 14]

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x = self.unet(images)   # [B*5, 14, H, W]
        x = self.conv(x)        # [B*5, 14, H, W]

        x = x.reshape(B, 5, 14, self.cfg.splat_size, self.cfg.splat_size)
        
        # # visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)    # [B, 5, splat_size, splat_size, 14] --> [B, N, 14]
        
        pos = self.pos_act(x[..., 0:3])     # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4]) # [B, N, 1]
        scale = self.scale_act(x[..., 4:7]) # [B, N, 3]
        rotation = self.rot_act(x[..., 7:11])   # [B, N, 3]
        rgbs = self.rgb_act(x[..., 11:])    # [B, N, 4]

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)    # [B, N, 14]
        return gaussians
    
    def forward(self, data, lambda_mse=1, lambda_lpips=1, lambda_top=1):
        # data: output of the dataloader
        # data = {
        #     [C, H, W]
        #     'input': ...,             (processed input images 5x9x256x256)
        #     'cam_poses_input': ...,   
        #     'images_output': ...,     (9x3x512x512)
        #     'masks_output': ...,      (.......)
        #     'cam_view_output': ...,          (colmap coordinate)
        #     'cam_view_proj_output': ...,     (colmap coordinate)
        #     'cam_pos_output': ...,           (colmap coordinate)
        # }
        # ------------
        # return: results = {
        #     'gaussians': ...,
        #     'images_pred': ...,
        #     'alphas_pred': ...,
        #     'loss_mse': ...,
        #     'loss_lpips': ...,
        #     'loss': ...,
        #     'psnr': ...,
        # }

        results = {}
        loss = 0

        images = data['input']  # [B, 5, 9, H, W], input features (not necessarily orthogonal)

        # predicting 3DGS representation
        gaussians = self.forward_gaussians(images)  # [B, N, 14]

        results['gaussians'] = gaussians

        # always use white background
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        # use the other views for rendering and supervision
        rendered_results = self.gs.render(gaussians, data['cam_view_output'], data['cam_view_proj_output'], data['cam_pos_output'], bg_color=bg_color)
        pred_images = rendered_results['image']  # [B, V, C, output_size, output_size]
        pred_alphas = rendered_results['alpha']  # [B, V, 1, output_size, output_size]
        pred_images = pred_images * pred_alphas + (1 - pred_alphas) * bg_color.view(1, 1, 3, 1, 1)

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output']   # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output']     # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + (1 - gt_masks) * bg_color.view(1, 1, 3, 1, 1)

        loss_mse_all = F.mse_loss(pred_images, gt_images) + self.cfg.lambda_alpha * F.mse_loss(pred_alphas, gt_masks)
        # loss_mse_top = F.mse_loss(pred_images[:, 4], gt_images[:, 4]) + self.cfg.lambda_alpha * F.mse_loss(pred_alphas[:, 4], gt_masks[:, 4])
        loss = loss + lambda_mse * (loss_mse_all) # + lambda_mse * (lambda_top - 1) * loss_mse_top

        if lambda_lpips > 0:
            loss_lpips_all = self.lpips_loss(
                # Rescale value from [0, 1] to [-1, -1] and resize to 256 to save memory cost
                F.interpolate(gt_images.view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                F.interpolate(pred_images.view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            # loss_lpips_top = self.lpips_loss(
            #     # Rescale value from [0, 1] to [-1, -1] and resize to 256 to save memory cost
            #     F.interpolate(gt_images[:, 4].view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            #     F.interpolate(pred_images[:, 4].view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            # ).mean()
            loss = loss + lambda_lpips * (loss_lpips_all) # + lambda_lpips * (lambda_top - 1) * loss_lpips_top

        results['loss'] = loss

        # metric
        with torch.no_grad():
            B, V, C, H, W = pred_images.shape

            # PSNR
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr
            
            # SSIM
            ssim = self.ssim_metric(pred_images.view(B * V, C, H, W), gt_images.view(B * V, C, H, W))
            results['ssim'] = ssim

            # LPIPS
            lpips = self.lpips_loss(
                F.interpolate(gt_images.view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                F.interpolate(pred_images.view(-1, 3, self.cfg.output_size, self.cfg.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['lpips'] = lpips

        return results