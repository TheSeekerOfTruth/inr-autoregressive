import os
from pathlib import Path
import torch
from torch.utils.data import Dataset

class INRDataset(Dataset):
    def __init__(self, folder_path, split='train', n_bins = 200):
        self.folder_path = Path(folder_path)
        self.split = 'train' if split == 'train' else 'test'
        self.split_dir = self.folder_path / ("training" if self.split == 'train' else "testing")
        self.file_list = [f for f in os.listdir(self.split_dir) if f.endswith('.pt')]
        self.n_bins = n_bins
        self.INIT_TOKEN = n_bins

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.split_dir / self.file_list[idx]
        data = torch.load(file_path, map_location='cpu')

        tokens = data['tokens'].long() 
        init_token_tensor = torch.tensor([self.INIT_TOKEN], dtype=torch.long)
        x_tokens = torch.cat([init_token_tensor, tokens[:-1]], dim=0)
        y_tokens = tokens
        

        return {
            'x_tokens': x_tokens,      
            'y_tokens': y_tokens,
            'layer_ids': data['layer_ids']
        }
