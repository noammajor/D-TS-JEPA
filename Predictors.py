import torch.nn as nn

class DiscreteJEPAPredictor(nn.Module):
    def __init__(
        self,
        num_patches,
        embed_dim,
        predictor_embed_dim,
        nhead,
        num_layers,
        mlp_ratio=4.0
    ):
        super().__init__()
        self.num_patches = num_patches
        self.predictor_embed_dim = predictor_embed_dim
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
        )
        embedder = PosEmbeder(dim=predictor_embed_dim, num_patches=num_patches)
        self.pos_embed = embedder.get_pos_embed(type='sine_cosine')   
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=nhead) for _ in range(num_layers)
        ])
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)

    def forward(self, context_vals, target_mask=None, context_pos=None):
        # context token is semantic or patch tokens
        B, ctx_size, _ = context_vals.shape
        x = self.predictor_embed(context_vals)
        if context_pos is not None:
            ctx_pe = self.pos_embed.repeat(B, 1, 1)
            ctx_pe = apply_mask(ctx_pe, context_pos) 
            x = x + ctx_pe
        if target_mask is not None:
            target_PE =self.pos_embed.repeat(B, 1, 1)
            target_PE = apply_mask(target_PE, target_mask)
            pred_tokens = self.mask_token.repeat(B, target_PE.size(1), 1) + target_PE
        
        for block in self.blocks:
            x = block(x)
        x = self.predictor_norm(x)

        if target_mask is not None:
            # S2P or P2P: We only care about the predicted masked areas
            x = x[:, ctx_size:, :]
        else:
            #P2S We predict all semantic tokens
            x = x[:, :self.num_semantic_tokens, :]
        return self.predictor_proj(x)
