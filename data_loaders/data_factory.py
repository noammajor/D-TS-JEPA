from torch.utils.data import DataLoader
from .data_class import CSVDataLoader, EvaluationDataLoader
import pandas as pd
import gzip
import numpy as np

import torch


def get_jepa_loaders(path, batch_size, ratio_patches=10, mask_ratio=0.9):
    dataloader = DataPuller()

    dataloader = DataLoader(dataloader,
                            batch_size=batch_size,
                            shuffle=True)

    return dataloader