import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.nn.init as init
import numpy as np

import einops
from einops.layers.torch import Rearrange
from einops import rearrange


class PatchEmbeddings(nn.Module):

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        channels: int 
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            Rearrange("b c h w -> b h w c")
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    
class PatchMerge(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (int): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, channel_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = channel_dim
        self.reduction = nn.Linear(4 * channel_dim, 2 * channel_dim, bias=False)
        self.norm = norm_layer(4 * channel_dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpand(nn.Module):
    def __init__(self, channel_dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = channel_dim
        self.expand = nn.Linear(channel_dim, 2* channel_dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(channel_dim // dim_scale)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = self.expand(x)
        B, H, W, C = x.shape

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B, H*2, W*2 ,C//4)
        x= self.norm(x)

        return x    

class FinalPatchExpand(nn.Module):
    def __init__(self, channel_dim, img_channels, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim_scale = dim_scale
        self.expand = nn.Linear(channel_dim, 16* channel_dim, bias=False)
        self.output_dim = channel_dim 
        self.norm = norm_layer(channel_dim)
        self.output = nn.Conv2d(in_channels=channel_dim,out_channels=img_channels ,kernel_size=1,bias=False)

    def forward(self, x):
        """
        x: B, H, W, C
        """

        x = self.expand(x)
        B, H, W, C = x.shape

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)
        
        x = x.view(B,H*self.dim_scale, W*self.dim_scale,-1)
        x = x.permute(0,3,1,2)
        x = self.output(x)

        return x
    

class MLPBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Mixer(nn.Module):

    def __init__(
        self,
        num_patches: int,
        num_channels: int
    ):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b h w c -> b c w h"),
            MLPBlock(num_patches, num_patches),
            Rearrange("b c w h -> b c h w"),
            MLPBlock(num_patches, num_patches),
            Rearrange("b c h w -> b h w c"),
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, num_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mixing(x)
        x = self.channel_mixing(x)
        return x


class Unet_Mixer_noSkip(nn.Module):

    def __init__(
        self,
        img_size: int = 256,
        img_channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 96,     
    ):    
        super().__init__()
        
        #mixer blocks
        self.mixer1= Mixer( img_size//4, embed_dim)
        self.mixer2= Mixer( img_size//8, embed_dim*2)
        self.mixer3= Mixer( img_size//16, embed_dim*4)
        self.mixer4= Mixer( img_size//32, embed_dim*8)
        
        self.mixer11= Mixer( img_size//4, embed_dim)
        self.mixer22= Mixer( img_size//8, embed_dim*2)
        self.mixer33= Mixer( img_size//16, embed_dim*4)
        self.mixer44= Mixer( img_size//32, embed_dim*8)
        
        
        
        #encode
        self.patch_embed = PatchEmbeddings(patch_size, embed_dim, img_channels)
        self.patch_merge1 = PatchMerge(embed_dim)
        self.patch_merge2= PatchMerge(embed_dim*2)
        self.patch_merge3= PatchMerge(embed_dim*4)
        
        #decode
        self.patch_expand1 = PatchExpand(embed_dim*8)
        self.patch_expand2 = PatchExpand(embed_dim*4)
        self.patch_expand3 = PatchExpand(embed_dim*2)
        self.final_expand = FinalPatchExpand(embed_dim, img_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y=[] 
        
        x = self.patch_embed(x)
        x = self.mixer1(x)
        x = self.mixer1(x)
        y.append(x)
        x = self.patch_merge1(x)
        x = self.mixer2(x)
        x = self.mixer2(x)
        y.append(x)
        x = self.patch_merge2(x)
        x = self.mixer3(x)
        x = self.mixer3(x)
        y.append(x)
        x = self.patch_merge3(x)
        
        x = self.mixer4(x)
        x = self.mixer4(x)
        x = self.mixer44(x)
        x = self.mixer44(x)
        
        x = self.patch_expand1(x)
        x = self.mixer33(x)
        x = self.mixer33(x) + y[2]
        x = self.patch_expand2(x)
        x = self.mixer22(x)
        x = self.mixer22(x) + y[1]
        x = self.patch_expand3(x)
        x = self.mixer11(x)
        x = self.mixer11(x) + y[0]
        x = self.final_expand(x)
        return x        