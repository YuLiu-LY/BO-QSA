import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from methods.utils import *
from timm.models.layers import DropPath
from methods.utils import PositionEmbed


class SlotAttention(nn.Module):
    def __init__(
        self,
        slot_size, 
        mlp_size, 
        truncate,
        epsilon=1.0,
        drop_path=0.2,
    ):
        super().__init__()
        self.slot_size = slot_size 
        self.epsilon = epsilon
        self.truncate = truncate

        self.norm_feature = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        
        self.project_k = linear(slot_size, slot_size, bias=False)
        self.project_v = linear(slot_size, slot_size, bias=False)
        self.project_q = linear(slot_size, slot_size, bias=False)

        self.gru = gru_cell(slot_size, slot_size)

        self.mlp = nn.Sequential(
            linear(slot_size, mlp_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_size, slot_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        print(self.truncate)
        assert self.truncate in ['bi-level', 'fixed-point', 'none']

    def forward(self, features, slots_init, num_iter=3):
        # `feature` has shape [batch_size, num_feature, inputs_size].
        features = self.norm_feature(features)  
        k = self.project_k(features)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features)  # Shape: [B, num_features, slot_size]

        B, N, D = v.shape
        slots = slots_init
        # Multiple rounds of attention.
        for i in range(num_iter):
            if i == num_iter - 1:
                if self.truncate == 'bi-level':
                    slots = slots.detach() + slots_init - slots_init.detach()
                elif self.truncate == 'fixed-point':
                    slots = slots.detach()
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)
            # Attention
            scale = D ** -0.5
            attn_logits= torch.einsum('bid,bjd->bij', q, k) * scale
            attn = F.softmax(attn_logits, dim=1)

            # Weighted mean
            attn_sum = torch.sum(attn, dim=-1, keepdim=True) + self.epsilon
            attn_wm = attn / attn_sum 
            updates = torch.einsum('bij, bjd->bid', attn_wm, v)            

            # Update slots
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            )
            slots = slots.reshape(B, -1, D)
            slots = slots + self.drop_path(self.mlp(self.norm_mlp(slots)))
        return slots, attn


class SlotAttentionEncoder(nn.Module):
    def __init__(
        self, num_iter, num_slots, feature_size, 
        slot_size, mlp_size, 
        resolution, truncate='bi-level', 
        init_method='embedding', drop_path=0.2):
        super().__init__()
        
        self.num_iter = num_iter
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.init_method = init_method

        self.pos_emb = PositionEmbed(feature_size, resolution)
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, slot_size)
        )

        self.slot_attention = SlotAttention(slot_size, mlp_size, truncate, drop_path=drop_path)

        assert init_method in ['shared_gaussian', 'embedding']
        if init_method == 'shared_gaussian':
            self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)
        elif init_method == 'embedding':
            self.slots_init = nn.Embedding(num_slots, slot_size)
            nn.init.xavier_uniform_(self.slots_init.weight)
        else:
            raise NotImplementedError
    
    def forward(self, f, sigma, slots_init=None):
        B = f.shape[0]
        f = self.pos_emb(f)
        f = torch.flatten(f, start_dim=2, end_dim=3).permute(0, 2, 1)
        f = self.mlp(f)

        if slots_init == None:
            # The first frame, initialize slots.
            if self.init_method == 'shared_gaussian':
                slots_init = torch.randn(B, self.num_slots, self.slot_size).type_as(f) * torch.exp(self.slot_log_sigma) + self.slot_mu
            elif self.init_method == 'embedding':
                mu = self.slots_init.weight.expand(B, -1, -1)
                z = torch.randn_like(mu).type_as(f)
                slots_init = mu + z * sigma * mu.detach()

        slots, attn  = self.slot_attention(f, slots_init, self.num_iter)
        
        return {
            'slots': slots,
            'slots_init': slots_init,
            'attn': attn,
        }




