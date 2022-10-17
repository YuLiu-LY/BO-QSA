import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from methods.utils import *


class dVAE(nn.Module):
    def __init__(self, vocab_size, img_channels, kernel_size):
        super().__init__()
        self.encoder = nn.Sequential(
            Conv2dBlock(img_channels, 64, 4, 4),
            Conv2dBlock(64, 64, kernel_size, 1, kernel_size // 2),
            Conv2dBlock(64, 64, kernel_size, 1, kernel_size // 2),
            Conv2dBlock(64, 64, kernel_size, 1, kernel_size // 2),
            Conv2dBlock(64, 64, kernel_size, 1, kernel_size // 2),
            Conv2dBlock(64, 64, kernel_size, 1, kernel_size // 2),
            Conv2dBlock(64, 64, kernel_size, 1, kernel_size // 2),
            conv2d(64, vocab_size, 1),
        )
            
        self.decoder = nn.Sequential(
            Conv2dBlock(vocab_size, 64, 1),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            conv2d(64, img_channels, 1),
        )

    def get_z(self, image):
        z_logits = F.log_softmax(self.encoder(image), dim=1)
        z = gumbel_softmax(z_logits, 0.1, True, dim=1).detach()
        return z 
    
    def forward(self, image, tau, return_z=False):
        # dvae encode
        z_logits = F.log_softmax(self.encoder(image), dim=1)
        z = gumbel_softmax(z_logits, tau, False, dim=1)
        # dvae recon
        recon = self.decoder(z)
        mse = ((image - recon) ** 2).sum() / image.shape[0]
        if return_z:
            z_hard = gumbel_softmax(z_logits, tau, True, dim=1)
            return mse, recon.clamp(0, 1), z_hard.detach()
        else:
            return mse, recon.clamp(0, 1)
    

