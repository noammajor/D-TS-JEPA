# Pre - Training
import time
import copy
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
from torchviz import make_dot
from Encoder import Encoder
from Predictors import DiscreteJEPAPredictor as Predictor
from data_loaders.data_puller import DataPuller
from mask_util import apply_mask
from data_loaders.data_factory import get_jepa_loaders
from config_files.config_pretrain import config
from main.utils import init_weights
from utils.modules import MLP, Block
from pos_embeder import PosEmbeder

def compute_discrete_jepa_loss(
    context_out, 
    target_out, 
    predictor, 
    masks, 
    non_masks,
    lambda_weights={'s2p': 1.0, 'p2s': 1.0, 'p2p': 1.0},
    beta_vq=1.0,
    current_global_step=0,
    total_training_steps=100000,
    vq_warmup = 0.15,
    batch_idx=0
    ):
    #warm up for VQ loss
    #vq_weight = min(1.0, current_global_step /int(vq_warmup * total_training_steps))
    z_s_target = target_out["quantized_semantic"]
    z_p_target = target_out["data_patches"]
    z_s_context = context_out["quantized_semantic"] 
    z_p_context = context_out["data_patches"]

    pred_s2p = predictor(z_s_context, target_mask=masks, task='S2P')
    l_s2p = F.mse_loss(pred_s2p, z_p_target.detach())

    pred_p2s = predictor(z_p_context, target_mask=None, task='P2S')
    l_p2s = F.mse_loss(pred_p2s, z_s_target.detach())
    with torch.no_grad():
            # 1. Is the Teacher actually producing diverse data?
            # If this is < 0.001, your Teacher has collapsed.
        t_var = z_s_target.std(dim=0).mean().item()
            
            # 2. Is the Student "cheating" by just mirroring the Teacher?
            # We calculate similarity between the prediction and the target.
        cos = torch.nn.CosineSimilarity(dim=-1)
        p2s_sim = cos(pred_p2s, z_s_target).mean().item()
            
            # 3. Information Leakage: Does the prediction look exactly like the first patch?
            # If this is > 0.9, your semantic tokens are just "copying" patch 0.
        leakage = cos(pred_p2s[:, 0, :], z_p_context[:, 0, :]).mean().item()
        if batch_idx % 10 == 0:
            print(f"\n--- Batch {batch_idx} Diagnostics ---")
            print(f"Shapes: Pred {pred_p2s.shape} | Target {z_s_target.shape}")
            print(f"Teacher Latent Var: {t_var:.6f} (Should be > 0.01)")
            print(f"P2S Similarity:     {p2s_sim:.4f} (If 0.99 + Low Loss = Cheating)")
            print(f"Input Leakage:      {leakage:.4f} (If high, attention is too direct)")
            print(f"Current P2S Loss:   {F.mse_loss(pred_p2s, z_s_target).item():.4f}")
            print("---------------------------------\n")

    pred_p2p = predictor(z_p_context, target_mask=masks, task='P2P')
    l_p2p = F.mse_loss(pred_p2p, z_p_target.detach())

    l_vq = context_out["vq_loss"]

    total_loss = (
        lambda_weights["S2P"] * l_s2p +
        lambda_weights["P2S"] * l_p2s +
        lambda_weights["P2P"] * l_p2p +
        beta_vq * l_vq
    )
    print(f"Losses => S2P: {l_s2p.item():.4f}, P2S: {l_p2s.item():.4f}, P2P: {l_p2p.item():.4f}, VQ: {l_vq.item():.4f}")
    return total_loss, {
        'l_s2p': l_s2p.item(),
        'l_p2s': l_p2s.item(),
        'l_p2p': l_p2p.item(),
        'l_vq': l_vq.item()
    }

def evaluate(encoder,
                predictor,
                dataloader,
                device,
                config,
                lambda_weights,
                beta_vq,
                current_global_step,
                total_training_steps,
                vq_warmup):
    encoder.eval()
    predictor.eval()
    val_loss = 0.0
    val_metrics = {'l_s2p': 0.0, 'l_p2s': 0.0, 'l_p2p': 0.0, 'l_vq': 0.0}
    with torch.no_grad():
        for patches, masks, non_masks in dataloader:
            patches, masks, non_masks = patches.to(device), masks.to(device), non_masks.to(device)
            target_out = encoder(patches)
            target_out = apply_mask(target_out, masks)
            
            context_out = encoder(patches, mask=non_masks)
            loss, loss_dict = compute_discrete_jepa_loss(
                context_out, 
                target_out, 
                predictor, 
                masks, 
                non_masks,
                lambda_weights=lambda_weights,
                beta_vq=beta_vq,
                current_global_step=current_global_step,
                total_training_steps=total_training_steps,
                vq_warmup=vq_warmup
            )
            val_loss += loss.item()
            for k, v in loss_dict.items():
                val_metrics[k] += v
    return val_loss / len(dataloader), {k: v / len(dataloader) for k, v in val_metrics.items()} 
    
def save_model(encoder, target_encoder, predictor, optimizer, epoch, path_save):
    checkpoint_dir = os.path.dirname(path_save)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Created directory: {checkpoint_dir}")
        except Exception as e:
            print(f"Could not create directory {checkpoint_dir}: {e}")
            return # Exit if we can't create the folder

    save_dict = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(), 
        "predictor": predictor.state_dict(),           
        "optimizer": optimizer.state_dict(),           
    }

    try:
        # 3. Save the file
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
    
    #config = prepare_args_pretrain(config)
    # Load Data
    train_dataset = DataPuller(
        data_paths=config["path_data"],
        patch_size=config["patch_size"],
        batch_size=config["batch_size"],
        ratio_patches=config["ratio_patches"],
        mask_ratio=config["mask_ratio"],
        masking_type=config["masking_type"],
        num_semantic_tokens=config["num_semantic_tokens"],
        input_variables=config["input_variables"],
        timestamp_cols=config["timestampcols"],
        type_data='train',
        val_prec=config["val_prec"],
        test_prec=config["test_prec"]
    )

    # Create Validation Dataset (Copy the master, change the pointer)
    val_dataset = copy.copy(train_dataset)
    val_dataset.which = 'val'

    # Create Test Dataset
    test_dataset = copy.copy(train_dataset)
    test_dataset.which = 'test'

    # Create Loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    input_dim = len(train_loader.dataset[0][0][0])
    
    encoder = Encoder(
        num_patches=len(train_loader.dataset[0][0]),
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
        num_patches=len(train_loader.dataset[0][0]),
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
    {"params": other_params, "lr": config["lr"], "weight_decay": config["weight_decay"]},
    {"params": codebook_params, "lr": config["codebook_lr"], "weight_decay": 0.0}
    ])
    steps_per_epoch = len(train_loader)
    total_steps = config["num_epochs"] * steps_per_epoch

    # mimicing the D-JEPA paper
    scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config["lr"],             # The peak learning rate from your config
    total_steps=total_steps,
    pct_start=0.05,                  # 5% warmup as per TD-JEPA
    anneal_strategy='cos',           # Cosine decay is standard used in D-JEPA]
    div_factor=10.0,                 # changed from 25 to 10
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
        * (0.999 - config["ema_momentum"])
        / (config["num_epochs"] * len(train_loader))
        for i in range(int(config["num_epochs"] *len(train_loader)) + 1)
    )
    num_batches = steps_per_epoch
    best_val_loss = float("inf")
    total_loss, total_var_encoder, total_var_decoder = 0.0, 0.0, 0.0
    save_model(encoder, encoder_ema, predictor,optimizer, 0, f"{path_save}_INITIAL")
    current_global_step = 0
    # Training Loop
    for epoch in range(config["num_epochs"]):
        print(f"Starting Epoch {epoch}/{config['num_epochs']}")
        encoder.train()
        predictor.train()
        running_loss = 0.0
        running_perplexity = 0.0

        for batch_idx, (patches, masks, non_masks) in enumerate(train_loader):
            optimizer.zero_grad()
            m = next(ema_scheduler)
            patches = patches.to(device)
            current_global_step += 1
            print(patches.shape)
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
                vq_warmup=config["vq_warmup"],
                batch_idx=batch_idx
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config["clip_grad"])
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), config["clip_grad"])
            
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for p, p_ema in zip(encoder.parameters(), encoder_ema.parameters()):
                    p_ema.copy_(m * p_ema + (1.0 - m) * p)
            
            running_loss += loss.item()
            running_perplexity += context_out["perplexity"].item()

        epoch_avg_loss = running_loss / len(train_loader)
        total_loss += epoch_avg_loss
        epoch_avg_perplexity = running_perplexity / len(train_loader)
        print(f"Epoch {epoch} completed. Avg Loss: {epoch_avg_loss:.4f}, Avg Perplexity: {epoch_avg_perplexity:.4f}")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.3g} - JEPA Loss: {total_loss:.4f},")

        if epoch % checkpoint_save == 0 and epoch != 0:
            save_model(encoder, epoch)
        val_loss, val_dict = evaluate(encoder, predictor, val_loader, device, config, config["lambda_weights"], config["beta_vq"], current_global_step, total_steps, config["vq_warmup"])
        
        print(f"Epoch {epoch} | Train Loss: {epoch_avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(encoder, encoder_ema, predictor, optimizer, epoch, f"{path_save}_BEST")
            print("New best validation loss! Model saved.")

    print("Training complete. Starting Final Test:")
    test_loss, test_dict = evaluate(encoder, predictor, test_loader, device, config, total_steps, total_steps)
    print(f"FINAL TEST RESULTS | Loss: {test_loss:.4f} | S2P: {test_dict['l_s2p']:.4f}")