import torch
import torch.nn as nn
import numpy as np
import math
from VQ import *
from mask_util import *
from utils.modules import *
from pos_embeder import PosEmbeder

class DiscreteJEPAPredictor(nn.Module):
    def __init__(
        self,
        num_patches,
        num_semantic_tokens,
        embed_dim,
        predictor_embed_dim,
        nhead,
        num_layers,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_semantic_tokens = num_semantic_tokens
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.proj_patch = nn.Linear(predictor_embed_dim, embed_dim)
        self.proj_semantic = nn.Linear(predictor_embed_dim, embed_dim)
        self.pos_embed_layer = PosEmbeder(dim=embed_dim, num_patches=num_patches) 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=nhead) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.semantic_query = nn.Parameter(torch.randn(1, num_semantic_tokens, predictor_embed_dim))

    def forward(self, x_input, target_mask=None, task='P2P'):
        """
        x_input: can be z_s_discrete or z_p depending on the task [cite: 133, 134]
        target_mask: Boolean mask of shape [B, num_patches] for target regions M [cite: 107]
        """
        B = x_input.shape[0]
        x = self.predictor_embed(x_input)
        
        if task == 'S2P' or task == 'P2P':
            B_orig = target_mask.shape[0]
            F = B // B_orig

            if F > 1:
                effective_mask = target_mask.unsqueeze(1).repeat(1, F, 1).view(B, -1)
            else:
                effective_mask = target_mask
            mask_idx = effective_mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
            full_pos_table = self.pos_embed_layer.pos_embed.expand(B, -1, -1)
            target_pos_embed = torch.gather(full_pos_table, dim=1, index=mask_idx)
            pred_tokens = self.mask_token + target_pos_embed
            x = torch.cat([x, pred_tokens], dim=1)
        if task == 'P2S':
            # Append semantic queries to the patch embeddings
            semantic_queries = self.semantic_query.expand(B, -1, -1)
            x = torch.cat([x, semantic_queries], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if task == 'S2P' or task == 'P2P':
            out = x[:, -target_pos_embed.size(1):, :]
            return self.proj_patch(out)
        
        elif task == 'P2S':
            out = x[:, -self.num_semantic_tokens:, :]
            return self.proj_semantic(out)