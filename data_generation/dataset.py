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

            coords = self.coordinate_grid(*img_2d.shape)
            values = img_2d.flatten().unsqueeze(-1)

            return coords, values, label

    def coordinate_grid(self, h=28, w=28):
        """Normalized (x, y) coordinate grid shared by every image of size (h, w).

        Identical for all MNIST images, so it can be computed once and reused
        instead of being rebuilt for every sample.
        """
        y = torch.linspace(0, 1, h)
        x = torch.linspace(0, 1, w)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    def get_target(self, idx):
        """Flattened pixel values (and label) for one sample, without rebuilding
        the coordinate grid. Use together with ``coordinate_grid``."""
        actual_idx = self.sorted_indices[idx]
        img, label = self.mnist[actual_idx]
        values = img.squeeze().flatten().unsqueeze(-1)
        return values, label

    def positions_by_label(self):
        """Map each digit label to the list of dataset positions (indices into
        this dataset) whose image has that label. Positions are contiguous and
        ordered because ``sorted_indices`` is sorted by label."""
        groups = {}
        for position, actual_idx in enumerate(self.sorted_indices):
            label = int(self.mnist.targets[actual_idx])
            groups.setdefault(label, []).append(position)
        return groups