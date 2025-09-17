import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial
from typing import Tuple, Literal

from core.attention import MemEffAttention


class MVAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_frames: int = 13,
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x):
        BV, C, H, W = x.shape
        B = BV // self.num_frames

        res = x
        x = self.norm(x)

        x = x.reshape(B, self.num_frames, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, -1, C)
        x = self.attn(x)
        x = x.reshape(B, self.num_frames, H, W, C).permute(0, 1, 4, 2, 3).contiguous().reshape(BV, C, H, W)

        if self.residual:
            x = (x + res) * self.skip_scale
        return x
    

class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()
 
        nets = []
        attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(in_channels, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for attn, net in zip(self.attns, self.nets):
            x = net(x)
            if attn:
                x = attn(x)

        if self.downsample:
            x = self.downsample(x)
  
        return x


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
        for i in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(in_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        
    def forward(self, x):
        x = self.nets[0](x)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn:
                x = attn(x)
            x = net(x)
        return x

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()

        self.upsample = None
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2.0, mode='nearest')

        nets = []
        attns = []
        
        layer_in_channels = in_channels + skip_channels
        
        for i in range(num_layers):
            cin = layer_in_channels if i == 0 else out_channels
            
            nets.append(ResnetBlock(cin, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

    def forward(self, x, skip_x):
        if self.upsample:
            x = self.upsample(x)
        
        x = torch.cat([x, skip_x], dim=1)

        for attn, net in zip(self.attns, self.nets):
            x = net(x)
            if attn:
                x = attn(x)
            
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 14,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024),
        down_attention: Tuple[bool, ...] = (False, False, False, True, True, True),
        mid_attention: bool = True,
        up_channels: Tuple[int, ...] = (1024, 1024, 512, 256, 128),
        up_attention: Tuple[bool, ...] = (True, True, True, False, False),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]
            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1),
                attention=down_attention[i],
                skip_scale=skip_scale,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.mid_block = MidBlock(down_channels[-1], attention=mid_attention, skip_scale=skip_scale)

        up_blocks = []
        cin = down_channels[-1]
        for i in range(len(up_channels)):
            skip_c = down_channels[max(0, len(down_channels) - 2 - i)]
            cout = up_channels[i]
            up_blocks.append(UpBlock(
                cin, skip_c, cout, 
                num_layers=layers_per_block,
                upsample=(i != len(up_channels) - 1),
                attention=up_attention[i],
                skip_scale=skip_scale,
            ))
            cin = cout
        self.up_blocks = nn.ModuleList(up_blocks)

        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.conv_in(x)

        skips = [x]
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
        
        x = self.mid_block(x)

        for i, block in enumerate(self.up_blocks):
            skip_connection = skips[len(self.down_blocks) - 1 - i]
            x = block(x, skip_connection)
            
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x