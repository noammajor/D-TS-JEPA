import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from making_style import get_mask_style

class DataPuller(Dataset):
    def __init__(self, 
    data_paths,
    patch_size,
    batch_size,
    ratio_patches,
    mask_ratio,
    masking_type,
    num_semantic_tokens,
    input_variables,
    timestamp_cols,
    type_data,
    val_prec = 0.1,
    test_prec = 0.25):
        self.batch_size = batch_size
        self.ratio_patches = ratio_patches
        self.mask_ratio = mask_ratio
        self.masking_type = masking_type
        self.num_semantic_tokens = num_semantic_tokens
        self.input_variables = input_variables
        self.timestamp_cols = timestamp_cols
        self.data_paths = data_paths
        self.val_prec = val_prec
        self.test_prec = test_prec
        self.which = type_data  # 'train', 'val', 'test'
        self.patch_size = patch_size
        self.chunk_size = self.patch_size * self.ratio_patches
        self.all_map = {'train': [], 'val': [], 'test': []}

        processed_dfs = []
        self.Train_Val_Test_splits = {
            'train': [],
            'val': [],
            'test': []
        }

        for path, t_col in zip(data_paths, timestamp_cols):
            df = pd.read_csv(path, parse_dates=[t_col], low_memory=False, sep=',')          
            fcols = df.select_dtypes("float").columns.tolist()
            df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
            processed_dfs.append(df)
            icols = df.select_dtypes("integer").columns
            df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")
            df.sort_values(by=[t_col], inplace=True)
            val_len = int(len(df) * self.val_prec)
            test_len = int(len(df) * self.test_prec)
            train_len = len(df) - val_len - test_len
            df = torch.tensor(df[self.input_variables].values).float()
            train_df, val_df, test_df = torch.split(df, [train_len, val_len, test_len])
            self.Train_Val_Test_splits['train'].append(train_df)
            self.Train_Val_Test_splits['val'].append(val_df)
            self.Train_Val_Test_splits['test'].append(test_df)
        
        for split_name in ['train', 'val', 'test']:
            for file_idx, tensor in enumerate(self.Train_Val_Test_splits[split_name]):
                num_chunks = (tensor.size(0) + self.chunk_size - 1) // self.chunk_size
                for chunk_idx in range(num_chunks):
                    self.all_map[split_name].append((file_idx, chunk_idx))

    def __len__(self):
        return len(self.all_map[self.which])
   
    def __getitem__(self, idx):
        file_idx, chunk_offset = self.all_map[self.which][idx]
        source_data = self.Train_Val_Test_splits[self.which][file_idx]
        start = chunk_offset * self.chunk_size
        end = start + self.chunk_size
        chunk = source_data[start:end]
        if chunk.size(0) < self.chunk_size:
            padding = torch.zeros((self.chunk_size - chunk.size(0), chunk.size(1)))
            chunk = torch.cat([chunk, padding], dim=0)
        patches_tensor = chunk.view(self.ratio_patches, self.patch_size, -1)
        context_idx, target_idx = get_mask_style(
            B=1, 
            num_patches=self.ratio_patches, 
            type=self.masking_type, 
            p=self.mask_ratio
        )
        return patches_tensor, context_idx.squeeze(0), target_idx.squeeze(0)

