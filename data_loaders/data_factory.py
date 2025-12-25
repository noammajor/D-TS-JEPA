import pdb

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,  Dataset_Neuro, Dataset_Saugeen_Web
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'neuro': Dataset_Neuro,
    'saugeen_web': Dataset_Saugeen_Web,
}


def get_jepa_loaders(path, batch_size, ratio_patches=10, mask_ratio=0.5, masking_type="bernoulli", num_semantic_tokens=4):
