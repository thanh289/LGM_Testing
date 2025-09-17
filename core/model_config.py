import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional

@dataclass
class Options:
    ### MODEL
    # Unet image input size
    input_size: int = 256
    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 64
    # gaussian render size
    output_size: int = 256

    vae_model: str = 'stabilityai/stable-diffusion-2-1-base'

    ### DATASET
    train_size: float = 0.8
    test_size: float = 0.1
    val_size: float = 0.1
    data_path: str = '/kaggle/input/objaverse-subset'
    # data mode (only support s3 now) 
    data_mode: Literal['s3'] = 's3'
    # Field of view in y direction of the dataset   
    fovy: float = 49.1
    # camera near plane (for clipping)    
    znear: float = 0.5
    # camera far plane (for clipping)   
    zfar: float = 2.5
    # number of total views
    num_views_total: int = 25
    # number of (input + test) views 
    # num_views_used: int = 17
    num_views_input: int = 13
    num_views_output: int = 13
    # camera radius (radius of camera orbitting around object)  
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 4

    ### TRAINING
    # workspace
    workspace: str = './workspace'
    # wandb
    wandb_key: str = 'd62c4d017c3c5c14173bc682fdaf504e1a8f2f3e'
    wandb_project_name: str ='LGM-4441-pretrain-ver9'
    wandb_experiment_name: str = 'lr=4e-4. lamdamse=1, lamdalpip=1, worker=4, accu=1, no*4gaussianmodel, no vae, newunet'
    wandb_experiment_id: str = None
    # fine-tuning
    fine_tune: bool = True
    # resume
    resume: Optional[str] = '/kaggle/input/original-checkpoint/model_fp16_fixrot.safetensors'
    # batch size (per-GPU)
    batch_size: int = 8
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 30
    # for mse loss
    lambda_alpha: float = 1.0
    lambda_top: float = 1.0     # lambda top_view loss 
    lambda_mse_start: float = 1.0
    lambda_mse_end: float = 1.0
    # lpips loss weight (loss = L_mse + lambda * L_lpips)
    lambda_lpips_start: float = 1
    lambda_lpips_end: float = 1
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    # mixed_precision: str = 'bf16'
    mixed_precision: str = 'fp16'
    # learning rate
    lr: float = 4e-4                                     
    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5

    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False


# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['lrm'] = 'the default settings for LGM'
config_defaults['lrm'] = Options()

config_doc['tiny'] = 'tiny model for ablation'
config_defaults['tiny'] = Options(
    input_size=256, 
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False, False),
    splat_size=64,
    output_size=256,
    batch_size=16,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['small'] = 'small model with lower resolution Gaussians'
config_defaults['small'] = Options(
    input_size=256,
    down_channels=(64, 128, 256, 512, 1024, 1024),
    down_attention=(False, False, False, True, True, True),
    up_channels=(1024, 1024, 512, 256),
    up_attention=(True, True, True, False),
    splat_size=64,
    output_size=256,
    batch_size=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['big'] = 'big model with higher resolution Gaussians'
config_defaults['big'] = Options(
    input_size=256,
    down_channels=(64, 128, 256, 512, 1024, 1024),
    down_attention=(False, False, False, True, True, True),
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=128,
    output_size=512, # render & supervise Gaussians at a higher resolution.
    batch_size=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)


config_doc['4441'] = 'settings for LGM without VAE'
config_defaults['4441'] = Options(

    input_size=128, 
    
    # 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 4
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    mid_attention=True,
    # 4 -> 8 -> 16 -> 32 -> 32
    up_channels=(512, 2566, 128, 64, 32),
    up_attention=(True, False, False, False, False),
    
    splat_size=32,
    output_size=512,
    batch_size=1, 
    gradient_accumulation_steps=1,
    num_workers = 4,
    lr=1e-4,
    num_epochs=63,
    mixed_precision='fp16',
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)


