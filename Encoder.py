# based on TS_JEPA: https://github.com/Sennadir/TS_JEPA/blob/main/src/models/encoder.py
import torch
import torch.nn as nn
import numpy as np
import math
from VQ import *
from mask_util import *
from utils.modules import *
from pos_embeder import PosEmbeder
from Tokenizer import TS_Tokenizer


class Encoder(nn.Module):
    def __init__(
        self,
        num_patches,
        num_semantic_tokens,
        dim_in,
        kernel_size,
        embed_dim,
        embed_bias,
        nhead,
        num_layers,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        jepa=False,
        embed_activation=nn.GELU(),
        codebook_size=512,
        commitment_cost=0.25
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_semantic_tokens = num_semantic_tokens
        self.activation = embed_activation if embed_activation else nn.GELU()
        
        #Semantic tokens
        self.semantic_tokens = nn.Parameter(torch.randn(1, self.num_semantic_tokens, embed_dim))
        torch.nn.init.trunc_normal_(self.semantic_tokens, std=0.02)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim), requires_grad=False
        )

        self.pos_embed_layer = PosEmbeder(dim=embed_dim, num_patches=num_patches)

        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=nhead, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, act_layer=nn.GELU, norm_layer=norm_layer
            ) for _ in range(num_layers)
        ])

        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.vector_quantizer = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=embed_dim,
            commitment_cost=commitment_cost
        )
        self.tokenizer = TS_Tokenizer(
            dim_in=dim_in,  
            kernel_size=kernel_size,
            embed_dim=embed_dim,
            embed_bias=embed_bias,
            activation=embed_activation
        )

        self.jepa = jepa

    def forward(self, x, mask=None):
        B, P, P_L , F = x.shape #[Batch, Patches, Patch_Len]
        #RevIN normalization
        self.mu = x.mean(dim=1, keepdim=True)
        self.sigma = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - self.mu) / self.sigma
        #channel independence
        x = x.permute(0, 3, 1, 2).reshape(B * F, P, P_L)
        #Encoder embedding
        x = self.tokenizer(x) 
        x = self.pos_embed_layer(x)
        if mask is not None:
            x = apply_mask(x, mask)  #[B, num_patches, D]
        sem_tokens = self.semantic_tokens.expand(B * F, -1, -1) #creates pointer copies of tokens for each example in the batch
        x = torch.cat((x, sem_tokens), dim=1) #[B, num_patches + num_semantic, D]

        #transformer blocks
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        out_semantic = x[:, -self.num_semantic_tokens:, :]
        data_patches = x[:, :-self.num_semantic_tokens, :]
        vq_loss, quantized_sem, perplexity, indices = self.vector_quantizer(out_semantic)
        return {
            "quantized_semantic": quantized_sem,
            "discrete_indices": indices,
            "data_patches": data_patches,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "orig_B": B,
            "orig_F": F
        }