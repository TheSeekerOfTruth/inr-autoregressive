import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class OptimizedDataset(Dataset):
    def __init__(self, root_dir='./data', train=True, transform=None, as_inr = True):
        super().__init__()
        self.as_inr = as_inr
        self.transform = transforms.ToTensor()
        self.mnist = datasets.MNIST(
            root = root_dir,
            train = train,
            transform=self.transform,
            download=False
        )
        self.sorted_indices = sorted(range(len(self.mnist)), key=lambda k: self.mnist[k][1])

    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
            actual_idx = self.sorted_indices[idx]
            img, label = self.mnist[actual_idx]
            
            if not self.as_inr:
                return img, label
            
            img_2d = img.squeeze() 
            h, w = img_2d.shape
            
            y = torch.linspace(0, 1, h)
            x = torch.linspace(0, 1, w)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            values = img_2d.flatten().unsqueeze(-1)
            
            return coords, values, label