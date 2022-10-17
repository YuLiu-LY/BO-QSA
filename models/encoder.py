import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from typing import Tuple
from torch import nn


class Encoder(nn.Module):
    def __init__(self, channels: Tuple[int, ...], strides: Tuple[int, ...], kernel_size):
        super().__init__()
        modules = []
        channel = 3
        for ch, s in zip(channels, strides):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(channel, ch, kernel_size, stride=s, padding=kernel_size//2),
                    nn.ReLU(),
                )
            )
            channel = ch
        self.conv = nn.Sequential(*modules)
        
    def forward(self, x):
        """
        input:
            x: input image, [B, 3, H, W]
        output:
            feature_map: [B, C, H_enc, W_enc]
        """
        x = self.conv(x)
        return x




