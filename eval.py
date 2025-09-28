from core.model_config import AllConfigs, Options
from core.model import LGM
from accelerate import Accelerator
from safetensors.torch import load_file
from core.dataset import ObjaverseDataset as Dataset
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR


import torch
import tyro
import kiui
import wandb
import numpy as np
import os
import random

def main():
    
    cfg = tyro.cli(AllConfigs)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )

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
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')

    val_dataset = Dataset(data_path=cfg.data_path, cfg=cfg, type='val')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )

    # accelerate
    model, val_dataloader = accelerator.prepare(
        model, val_dataloader
    )

    if not cfg.fine_tune and cfg.resume is not None:
        # NOTE: cfg.resume (dir type) must be saved by accelerator.save_state()
        # Continue training by loading all state of optimizer, model, scheduler
        accelerator.load_state(cfg.resume, strict=False)

    if accelerator.is_main_process:
        accelerator.print(f'[INFO] start evaluation for {len(val_dataset)} objects...')
    # eval
    with torch.no_grad():
        model.eval()
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        if accelerator.is_main_process:
            pbar2 = tqdm(val_dataloader, desc=f"[Evaluation]")

        for i, data in enumerate(val_dataloader):
            out = model(data)

            psnr = out['psnr']
            ssim = out['ssim']
            lpips = out['lpips']
            total_psnr += psnr.detach()
            total_ssim += ssim.detach()
            total_lpips += lpips.detach()

            if accelerator.is_main_process:
                pbar2.update(1)
                if i % 5 == 0:
                    gt_images = data['images_output'].detach().cpu().numpy()    # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3)
                    kiui.utils.write_image(f'{cfg.workspace}/{i}_eval_gt_images.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy()     # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.utils.write_image(f'{cfg.workspace}/{i}_eval_pred_images.jpg', pred_images)

        if accelerator.is_main_process:
            pbar2.close()
        torch.cuda.empty_cache()

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        total_ssim = accelerator.gather_for_metrics(total_ssim).mean()
        total_lpips = accelerator.gather_for_metrics(total_lpips).mean()
        if accelerator.is_main_process:
            total_psnr /= len(val_dataloader)
            total_ssim /= len(val_dataloader)
            total_lpips /= len(val_dataloader)
            accelerator.print(f'[EVAL] psnr: {total_psnr:.4f}, ssim: {total_ssim:.4f}, lpips: {total_lpips:.4f}')


if __name__ == "__main__":
    # 1. Seed everything
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Force deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)

    main()