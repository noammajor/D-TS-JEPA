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
        embedder = PosEmbeder(dim=predictor_embed_dim, num_patches=num_patches)
        self.pos_embed = embedder.get_pos_embed(type='sine_cosine')   
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=nhead) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(predictor_embed_dim)

    def forward(self, x_input, target_mask=None, task='P2P'):
        """
        x_input: can be z_s_discrete or z_p depending on the task [cite: 133, 134]
        target_mask: Boolean mask of shape [B, num_patches] for target regions M [cite: 107]
        """
        B = x_input.shape[0]
        x = self.predictor_embed(x_input)
        
        if task == 'S2P' or task == 'P2P':
            target_pos_embed = self.pos_embed.repeat(B, 1, 1)[target_mask].view(B, -1, x.size(-1))
            
            # Combine context with mask tokens + target positions [cite: 133, 156]
            pred_tokens = self.mask_token + target_pos_embed
            x = torch.cat([x, pred_tokens], dim=1)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if task == 'S2P' or task == 'P2P':
            out = x[:, -target_pos_embed.size(1):, :]
            return self.proj_patch(out)
        
        elif task == 'P2S':
            out = x[:, :self.num_semantic_tokens, :]
            return self.proj_semantic(out)