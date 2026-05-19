import os
from pathlib import Path
import torch
from torch.utils.data import Dataset

class INRDataset(Dataset):
    def __init__(self, folder_path, split='train'):
        self.folder_path = Path(folder_path)
        self.split = 'train' if split == 'train' else 'test'
        self.split_dir = self.folder_path / ("training" if self.split == 'train' else "testing")
        self.file_list = [f for f in os.listdir(self.split_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.split_dir / self.file_list[idx]
        data = torch.load(file_path, map_location='cpu')

        neurons = data['neurons'] 
        layer_ids = data['layer_ids'] 

        x_neurons = neurons[:-1]
        y_neurons = neurons[1:]
        
        l_ids = layer_ids[:-1]
        target_l_ids = layer_ids[1:]

        return {
            'x_neurons': x_neurons,      
            'y_neurons': y_neurons,      
            'layer_ids': l_ids,          
            'target_layer_ids': target_l_ids 
        }
