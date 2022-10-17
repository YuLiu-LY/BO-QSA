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
from methods.utils import PositionEmbed


class SlotAttention(nn.Module):
    def __init__(
        self,
        slot_size, 
        mlp_size, 
        epsilon=1e-8,
    ):
        super().__init__()
        self.slot_size = slot_size 
        self.epsilon = epsilon

        self.norm_feature = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        
        self.project_k = nn.Linear(slot_size, slot_size, bias=False)
        self.project_v = nn.Linear(slot_size, slot_size, bias=False)
        self.project_q = nn.Linear(slot_size, slot_size, bias=False)

        self.gru = nn.GRUCell(slot_size, slot_size)

        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, slot_size),
        )

    def forward(self, features, slots_init, num_iter=3):
        # `feature` has shape [batch_size, num_feature, inputs_size].
        features = self.norm_feature(features)  
        k = self.project_k(features)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features)  # Shape: [B, num_features, slot_size]

        B, N, D = v.shape
        slots = slots_init
        # Multiple rounds of attention.
        for i in range(num_iter):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)
            # Attention
            scale = D ** -0.5
            attn_logits= torch.einsum('bid,bjd->bij', q, k) * scale
            attn = F.softmax(attn_logits, dim=1)

            # Weighted mean
            attn = attn + self.epsilon
            attn_sum = torch.sum(attn, dim=-1, keepdim=True)
            attn_wm = attn / attn_sum 
            updates = torch.einsum('bij, bjd->bid', attn_wm, v)            

            # Update slots
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            )
            slots = slots.reshape(B, -1, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn


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
            kernel_size=5, 
            channels=args.decoder_channels,
            strides=args.decoder_strides,
        )

        encoder_stride = 1
        for s in args.encoder_strides:
            encoder_stride *= s
        
        feature_size = args.encoder_channels[-1]
        resolution = [args.resolution[0] // encoder_stride, args.resolution[1] // encoder_stride]
        self.pos_emb = PositionEmbed(feature_size, resolution)
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, args.mlp_size),
            nn.ReLU(),
            nn.Linear(args.mlp_size, args.slot_size)
        )

        self.slot_attn = SlotAttention(args.slot_size, args.mlp_size)
        self.slot_mu = nn.Parameter(torch.zeros(1, args.num_slots, args.slot_size))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, args.num_slots, args.slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        self.num_iter = args.num_iter
        self.num_slots = args.num_slots
        self.slot_size = args.slot_size

    def forward(self, x, sigma=0):
        B, _, _, _ = x.shape
        features = self.encoder(x)
        f = self.pos_emb(features)
        f = torch.flatten(f, start_dim=2, end_dim=3).permute(0, 2, 1)
        f = self.mlp(f)

        z = torch.randn(B, self.num_slots, self.slot_size).type_as(x)
        slots_init = self.slot_mu + z * self.slot_log_sigma.exp()

        slots, attn = self.slot_attn(f, slots_init, self.num_iter)
        masks, recons = self.decoder(slots)

        recon = torch.sum(recons * masks, dim=1)
        mse = F.mse_loss(recon, x)

        return {
            "mse": mse,
            "recon": recon, 
            "recons": recons, 
            "masks": masks,
            # "attn": attn,
            # 'slots': slots
        }


   