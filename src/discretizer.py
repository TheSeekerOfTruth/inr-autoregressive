# we use this rather crude method, because it worked better than using quantiles.

import torch


def bin_quantize_tensor(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Discretize x to n_bins evenly-spaced levels in [-0.5, 0.5]."""
    if n_bins == 1:
        return torch.zeros_like(x)
    idx = ((x + 0.5) * (n_bins - 1)).round().clamp(0, n_bins - 1)
    return idx / (n_bins - 1) - 0.5


def discretize(neuron_tokens: list, n_bins: int = 100) -> list:
    """
    Quantize tokenizer output to n_bins levels in [-0.5, 0.5].

    Args:
        neuron_tokens: output of INRTokenizer.tokenize — List[List[Tensor]]
        n_bins: number of discrete levels (default 100)

    Returns:
        Same structure with quantized tensors.
    """
    return [
        [bin_quantize_tensor(token, n_bins) for token in layer]
        for layer in neuron_tokens
    ]
