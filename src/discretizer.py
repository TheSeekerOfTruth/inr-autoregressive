import torch
import numpy as np


class INRDiscretizer:
    def __init__(self, mode='uniform', n_bins=50, k=50, min_val=-0.5, max_val=0.5, codebooks=None):
        self.mode = mode
        self.n_bins = n_bins
        self.k = k
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val
        self.codebooks = codebooks

    def discretize(self, list_of_neuron_tokens: list) -> list:
        """Entry point for both discretization modes."""
        if self.mode == 'uniform':
            return [[self._quantize_uniform(t) for t in layer] for layer in list_of_neuron_tokens]
        else:
            return [[self._quantize_kmeans(t) for t in layer] for layer in list_of_neuron_tokens]

    def _quantize_uniform(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.min_val) / self.range
        idx = (normalized * (self.n_bins - 1)).round().clamp(0, self.n_bins - 1)
        return idx.long()

    def _quantize_kmeans(self, token: torch.Tensor) -> torch.Tensor:
        dim = token.shape[0]
        # Calculate L2 distance to centroids and take argmin
        dist = torch.norm(self.codebooks[dim] - token, dim=1)
        if(dim == 3):
            return torch.argmin(dist).long()
        else:
            return torch.argmin(dist).long() + self.codebooks[dim].size(0)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
            """
            Maps global indices back to continuous tensors.
            indices: 1D Tensor of global IDs.
            """
            if self.mode == 'uniform':
                vocab = self.get_vocabulary()
                return vocab[indices]

            offset = self.codebooks[3].size(0)
            
            max_dim = max(dim for dim in self.codebooks.keys())
            output = torch.zeros(indices.size(0), max_dim)
            print(output.shape)
            mask_3 = indices < offset
            output[mask_3, :3] = self.codebooks[3][indices[mask_3]]
            mask_33 = ~mask_3
            relative_indices = indices[mask_33] - offset
            output[mask_33, :33] = self.codebooks[33][relative_indices]

            return output

    def get_vocabulary(self) -> torch.Tensor:
        """Returns the uniform codebook."""
        steps = torch.arange(0, self.n_bins, dtype=torch.float32)
        return (steps / (self.n_bins - 1)) * self.range + self.min_val