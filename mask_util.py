import torch

def apply_mask(x, masks, type = "bernoulli", p = 0.5):
    if type == "bernoulli":
        return apply_bernoulli_mask(x, p)
    elif type == "block":
        return apply_block_mask(x)




def apply_bernoulli_mask(x, p):
    B, L, D = x.shape
    device = x.device

    mask = torch.rand(B, L, device=device) < p 
    expanded_mask = mask.unsqueeze(-1)
    masked_x = x.masked_fill(expanded_mask, 0.0)
    return masked_x, mask

def apply_block_mask(x, block_size=10):
    B, L, D = x.shape
    mask = torch.zeros(B, L, device=x.device, dtype=torch.bool)
    
    for i in range(B):
        start = torch.randint(0, L - block_size, (1,))
        mask[i, start : start + block_size] = True
        
    masked_x = x.masked_fill(mask.unsqueeze(-1), 0.0)
    return masked_x, mask

