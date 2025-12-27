config = {

    #learning rates
    "lr": 6e-5,
    "end_lr": 1e-4,
    "num_epochs": 5001,
    "ema_momentum" : 0.998,
    "codebook_lr" : 1e-5,
    "weight_decay" : 5e-2,

    #masking
    "mask_ratio" : 0.25,
    "masking_type" : "block",

    #encoder
    "num_semantic_tokens" : 4,
    "encoder_embed_dim" : 128,
    "nhead" : 8,
    "num_encoder_layers" : 4,
    "mlp_ratio" : 4.0,
    "qkv_bias" : True,
    "qk_scale" : None,
    "drop_rate" : 0.1,
    "attn_drop_rate" : 0.1,
    "kernel_size" : 3,
    "embed_bias" : True,
    "codebook_size" : 512,
    "commitment_cost" : 0.25,
    "patch_size": 8,

    #predictor
    "predictor_embed_dim": 64,
    "predictor_nhead" : 4,
    "predictor_num_layers" : 2,

    #data
    "checkpoint_save" : 5000,
    "checkpoint_print": 30,
    "path_save" : "./output_model/",
    "ratio_patches" : 12,
    "batch_size" : 32,

    # Loader
    "clip_grad": 10,
    "warmup_ratio": 0.12,
    "ipe_scale": 1.25,

    #weights for loss terms
    "lambda_weights" : {
        "P2P": 1.0,
        "S2P": 1.0,
        "P2S": 1.0,
    },
    "beta_vq" : 0.01,
    "vq_warmup": 0.15,

    #data paths
    "timestampcols" : ['date'],
    "input_variables" : ['OT' , "T (degC)", "Tpot (K)","Tdew (degC)","rh (%)"],
    "val_prec": 0.1,
    "test_prec": 0.25,
    "path_data" : ['./data/weather.csv'],

}