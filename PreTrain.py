# Pre - Training
import time
import copy
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from Encoder import Encoder
from Predictors import Predictor
from mask_utils import apply_mask
from src.data_loaders.data_loader import get_jepa_loaders
from config.config_pretrain import config
from main.utils import init_weights

def compute_discrete_jepa_loss(
    context_out, 
    target_out, 
    predictor, 
    masks, 
    non_masks, # Add this to help predictor with context positions
    lambda_weights={'s2p': 1.0, 'p2s': 1.0, 'p2p': 1.0},
    beta_vq=1.0,
    current_global_step=0,
    total_training_steps=100000,
    vq_warmup = 0.15
    ):
    #warm up for VQ loss
    vq_weight = min(1.0, current_global_step /int(vq_warmup * total_training_steps))
    z_s_target = target_out["quantized_semantic"]
    z_p_target = target_out["data_patches"]

    z_s_context = context_out["quantized_semantic"] 
    z_p_context = context_out["data_patches"]

    mask_idx = masks.unsqueeze(-1).expand(-1, -1, z_p_target.size(-1))
    target_p_masked = torch.gather(z_p_target, dim=1, index=mask_idx) # [B, M, D]

    pred_s2p = predictor(z_s_context, target_mask=masks, task='S2P')
    l_s2p = F.mse_loss(pred_s2p, target_p_masked)

    pred_p2s = predictor(z_p_context, non_masks=non_masks, task='P2S')
    l_p2s = F.mse_loss(pred_p2s, z_s_target)

    pred_p2p = predictor(z_p_context, target_mask=masks, non_masks=non_masks, task='P2P')
    l_p2p = F.mse_loss(pred_p2p, target_p_masked)

    l_vq = context_out["vq_loss"]

    total_loss = (
        lambda_weights['S2P'] * l_s2p +
        lambda_weights['P2S'] * l_p2s +
        lambda_weights['P2P'] * l_p2p +
        vq_weight*beta_vq * l_vq
    )
    
    return total_loss, {
        'l_s2p': l_s2p.item(),
        'l_p2s': l_p2s.item(),
        'l_p2p': l_p2p.item(),
        'l_vq': l_vq.item()
    }
    
def save_model(encoder, target_encoder, predictor, optimizer, epoch, path_save):
    save_dict = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(), # The EMA Teacher 
        "predictor": predictor.state_dict(),           # The 3-head predictor [cite: 118, 133]
        "optimizer": optimizer.state_dict(),           # Critical for resuming training
    }

    try:
        path_name = f"{path_save}_epoch_{epoch}.pt"
        torch.save(save_dict, path_name)
        print(f"Checkpoint saved: {path_name}")
    except Exception as e:
        print(f"Problem saving checkpoint: {e}")

# we can use diffrent one - this is TS_JEPA basic
def lr_lambda(epoch):
    start_lr = config["lr"]
    end_lr = config["end_lr"]
    if epoch < config["num_epochs"]:
        return start_lr + (end_lr - start_lr) * (epoch / (config["num_epochs"] - 1))
    else:
        return end_lr

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = prepare_args_pretrain(config)
    # Load Data
    loader = get_jepa_loaders(
        config["path_data"],
        config["batch_size"],
        config["ratio_patches"],
        config["mask_ratio"],
        config["masking_type"],
        config["num_semantic_tokens"]
    )
    input_dim = len(loader.dataset[0][0][0])
    encoder = Encoder(
        num_patches=len(loader.dataset[0][0]),
        num_semantic_tokens=config["num_semantic_tokens"],
        dim_in=input_dim,
        kernel_size=config["kernel_size"],
        embed_dim=config["encoder_embed_dim"],
        embed_bias=config["embed_bias"],
        nhead=config["nhead"],
        num_layers=config["num_encoder_layers"],
        mlp_ratio=config["mlp_ratio"],
        qkv_bias=config["qkv_bias"],
        qk_scale=config["qk_scale"],
        drop_rate=config["drop_rate"],
        attn_drop_rate=config["attn_drop_rate"],
        norm_layer=torch.nn.LayerNorm,
        jepa=True,
        embed_activation=torch.nn.GELU(),
        codebook_size=config["codebook_size"],
        commitment_cost=config["commitment_cost"]
    )
    predictor = Predictor(
        num_patches=len(loader.dataset[0][0]),
        num_semantic_tokens=config["num_semantic_tokens"],
        embed_dim=config["encoder_embed_dim"],
        predictor_embed_dim=config["predictor_embed_dim"],
        nhead=config["predictor_nhead"],
        num_layers=config["predictor_num_layers"],
    )

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    codebook_params = [p for n, p in encoder.named_parameters() if "embedding" in n]
    other_params = [p for n, p in encoder.named_parameters() if "embedding" not in n]
    other_params += list(predictor.parameters())

    optimizer = torch.optim.AdamW([
    {"params": other_params, "lr": config["lr"]},
    {"params": codebook_params, "lr": config["codebook_lr"]}  
    ])
    steps_per_epoch = len(loader)
    total_steps = config["num_epochs"] * steps_per_epoch

    # mimicing the D-JEPA paper
    scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config["lr"],             # The peak learning rate from your config
    total_steps=total_steps,
    pct_start=0.05,                  # 5% warmup as per TD-JEPA
    anneal_strategy='cos',           # Cosine decay is standard used in D-JEPA]
    div_factor=25.0,                 # defualt = Initial lr = max_lr / 25
    final_div_factor=1e4             # defualt
    )
    encoder = encoder.to(device)
    predictor = predictor.to(device)

    # Initialize the Target Encoder
    encoder_ema = copy.deepcopy(encoder)

    for p in encoder_ema.parameters():
        p.requires_grad = False
    
    checkpoint_save = config["checkpoint_save"]
    checkpoint_print = config["checkpoint_print"]
    path_save = config["path_save"]
    clip_grad = config["clip_grad"]
    warmup = config["warmup_ratio"] * config["num_epochs"]

    ema_scheduler = (
        config["ema_momentum"]
        + i
        * (1 - config["ema_momentum"])
        / (config["num_epochs"] * config["ipe_scale"])
        for i in range(int(config["num_epochs"] * config["ipe_scale"]) + 1)
    )
    num_batches = len(loader)
    total_loss, total_var_encoder, total_var_decoder = 0.0, 0.0, 0.0
    save_model(encoder, 0)
    current_global_step = 0
    # Training Loop
    for epoch in range(config["num_epochs"]):
        encoder.train()
        predictor.train()
        running_loss = 0.0
        running_perplexity = 0.0

        for patches, masks, non_masks in loader:
            optimizer.zero_grad()
            m = next(ema_scheduler)
            patches = patches.to(device)
            current_global_step += 1

            #channel independence:
            B, L, C = patches.shape
            with torch.no_grad():
                target_out = encoder_ema(patches)
                target_out["data_patches"] = F.layer_norm(
                    target_out["data_patches"], 
                    (target_out["data_patches"].size(-1),)
                )
                target_out = apply_mask(target_out, masks)
                z_s_target = target_out["quantized_semantic"] 
                #patches
                z_p_target = target_out["data_patches"]

            
            context_out = encoder(patches, mask=non_masks)
            loss, loss_dict = compute_discrete_jepa_loss(
                context_out, 
                target_out, 
                predictor, 
                masks, 
                non_masks,
                lambda_weights=config["lambda_weights"],
                beta_vq=config["beta_vq"],
                current_global_step=current_global_step,
                total_training_steps=total_steps,
                vq_warmup=config["vq_warmup"]
            )
            loss.backward()
        
            # Gradient clipping is vital for VQ-based models
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config["clip_grad"])
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), config["clip_grad"])
            
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for p, p_ema in zip(encoder.parameters(), encoder_ema.parameters()):
                    p_ema.copy_(m * p_ema + (1.0 - m) * p)
            
            running_loss += loss.item()
            running_perplexity += context_out["perplexity"].item()

        
        epoch_avg_loss = running_loss / len(loader)
        epoch_avg_perplexity = running_perplexity / len(loader)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.3g} - JEPA Loss: {total_loss:.4f},")

        if epoch % checkpoint_save == 0 and epoch != 0:
            save_model(encoder, epoch)