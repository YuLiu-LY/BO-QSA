import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import torch
from torch import nn
from torch.nn import functional as F

from models.encoder import Encoder
from models.mixture_decoder import Decoder
from models.slot_attn import SlotAttentionEncoder
from sklearn.cluster import KMeans


class SlotAttentionModel(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.encoder = Encoder(
            channels=args.encoder_channels, 
            strides=args.encoder_strides, 
            kernel_size=args.encoder_kernel_size,
        )
        self.decoder = Decoder(
            resolution=args.resolution, 
            init_resolution=args.init_resolution, 
            slot_size=args.slot_size, 
            kernel_size=args.decoder_kernel_size,
            channels=args.decoder_channels,
            strides=args.decoder_strides,
        )

        encoder_stride = 1
        for s in args.encoder_strides:
            encoder_stride *= s

        self.slot_attn = SlotAttentionEncoder(
            args.num_iter, args.num_slots, args.encoder_channels[-1], 
            args.slot_size, args.mlp_size, 
            [args.resolution[0] // encoder_stride, args.resolution[1] // encoder_stride], 
            args.truncate, args.init_method)

        self.num_iter = args.num_iter
        self.num_slots = args.num_slots
        self.slot_size = args.slot_size

        self.use_post_cluster = args.use_post_cluster
        self.lambda_c = args.lambda_c
        if self.use_post_cluster:
            self.register_buffer('post_cluster', torch.zeros(1, args.num_slots, args.slot_size))
            nn.init.xavier_normal_(self.post_cluster)
        self.kmeans = KMeans(n_clusters=args.num_slots, random_state=args.seed) if args.use_kmeans else None

    def forward(self, x, sigma=0, is_Train=False):
        B = x.shape[0]
        f = self.encoder(x)
        if self.use_post_cluster:
            slots_init = self.post_cluster.repeat(B, 1, 1)
            slot_attn_out = self.slot_attn(f, sigma=sigma, slots_init=slots_init)
            slots = slot_attn_out['slots']
            if is_Train:
                # update post cluster, shape: 1 x num_slots x slot_size
                if self.kmeans is not None:
                    print('kmeans')
                    self.kmeans.fit(slots.detach().reshape(-1, self.slot_size).cpu().numpy())
                    update = torch.Tensor(self.kmeans.cluster_centers_.reshape(1, self.num_slots, self.slot_size)).to(x.device)
                    self.post_cluster = self.lambda_c * update + (1 - self.lambda_c) * self.post_cluster
                else:
                    update = slots.detach().mean(dim=0, keepdim=True)
                    self.post_cluster = self.lambda_c * update + (1 - self.lambda_c) * self.post_cluster
        else:
            slot_attn_out = self.slot_attn(f, sigma=sigma)
            slots = slot_attn_out['slots']
        masks, recons = self.decoder(slots)

        recon = torch.sum(recons * masks, dim=1)
        mse = F.mse_loss(recon, x)

        return {
            "mse": mse,
            "recon": recon, 
            "recons": recons, 
            "masks": masks,
            # "attns": slot_attn_out['attn'],
            # 'slots': slots
        }


   