import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor

class ReconNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def pad(self, x):
        _, _, h, w = x.shape
        hp, wp = self.net.patch_size
        f1 = ( (wp - w % wp) % wp ) / 2
        f2 = ( (hp - h % hp) % hp ) / 2
        wpad = [floor(f1), ceil(f1)]
        hpad = [floor(f2), ceil(f2)]
        x = F.pad(x, wpad+hpad)
        
        return x, wpad, hpad
    
    def unpad(self, x, wpad, hpad):
        
        return x[..., hpad[0] : x.shape[-2]-hpad[1], wpad[0] : x.shape[-1]-wpad[1]]       
        
    def forward(self, x, k=None):     
        x, wpad, hpad = self.pad(x)
        x = self.net(x, k)
        x = self.unpad(x, wpad, hpad)

        return x