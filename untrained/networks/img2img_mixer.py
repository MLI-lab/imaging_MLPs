import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.init as init


import einops
from einops.layers.torch import Rearrange
from einops import rearrange


class PatchEmbedding(nn.Module):

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
    

class PatchExpansion(nn.Module):
    def __init__(self, dim_scale, channel_dim, img_channels, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim_scale = dim_scale
        self.expand = nn.Linear(channel_dim, dim_scale**2* channel_dim, bias=False)
        self.output_dim = channel_dim 
        self.norm = norm_layer(channel_dim)
        self.output = nn.Conv2d(in_channels=channel_dim,out_channels=img_channels ,kernel_size=1,bias=False)

    def forward(self, x):

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
        num_channels: int,
        f_hidden: int
    ):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b h w c -> b c w h"),
            MLPBlock(num_patches, num_patches*f_hidden),
            Rearrange("b c w h -> b c h w"),
            MLPBlock(num_patches, num_patches*f_hidden),
            Rearrange("b c h w -> b h w c"),
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, num_channels*f_hidden)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x


class Img2Img_Mixer(nn.Module):

    def __init__(
        self,
        img_size: int = 256,
        img_channels: int = 1,
        patch_size: int = 4,
        embed_dim: int = 128,
        num_layers: int = 16,
        f_hidden: int = 8,
    ):    
        super().__init__()
        
        self.patch_embed = PatchEmbedding(patch_size, embed_dim, img_channels)
        
        layers = [ Mixer(img_size//patch_size, embed_dim, f_hidden)
            for _ in range(num_layers)]
        self.mixer_layers = nn.Sequential(*layers)

        self.patch_expand = PatchExpansion(patch_size, embed_dim, img_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                
        x = self.patch_embed(x)
        x = self.mixer_layers(x)
        x = self.patch_expand(x)
        
        return x        