# we use this rather crude method, because it worked better than using quantiles.

import torch

class INRDiscretizer:
    """Handles the uniform discretization of continuous INR weight tensors into a fixed vocabulary."""
    def __init__(self, n_bins: int = 100, min_val: float = -0.5, max_val: float = 0.5):
        self.n_bins = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val

    def quantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Discretize a single tensor to n_bins evenly-spaced levels in [min_val, max_val]."""
        if self.n_bins == 1:
            return torch.zeros_like(x)

        # Shift to 0.0 - 1.0 based on dynamic min_val
        normalized = (x - self.min_val) / self.range

        # Scale to index range, round, and clamp
        max_idx = self.n_bins - 1
        idx = (normalized * max_idx).round().clamp(0, max_idx)
        return idx.long()

    def discretize_tokens(self, neuron_tokens: list) -> list:
        """Quantize a nested list structure of token tensors."""
        return [
            [self.quantize_tensor(token) for token in layer]
            for layer in neuron_tokens
        ]

    def get_vocabulary(self) -> torch.Tensor:
        """Returns the explicit codebook/vocabulary of possible quantized values."""
        if self.n_bins == 1:
            return torch.tensor([0.0], dtype=torch.float32)
        max_idx = self.n_bins - 1
        steps = torch.arange(0, self.n_bins, dtype=torch.float32)
        return (steps / max_idx) * self.range + self.min_val
