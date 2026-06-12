import torch
import numpy as np
from sklearn.cluster import KMeans
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
        return torch.argmin(dist).long()

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
            """
            Maps global indices back to continuous tensors.
            indices: 1D Tensor of global IDs.
            """
            if(self.mode=="uniform"):
                return self.get_vocabulary()[indices[indices != 201]]
            else:
                return self.codebooks[9][indices]

    def get_vocabulary(self) -> torch.Tensor:
        """Returns the uniform codebook."""
        steps = torch.arange(0, self.n_bins, dtype=torch.float32)
        return (steps / (self.n_bins - 1)) * self.range + self.min_val

def kmeans_discretize_zero_padding(list_of_neuron_tokens: list, k: int = 1000) -> list:
    """
    Quantize neurons from multiple INRs using a shared KMeans codebook.

    Neurons across layers may differ in dimension (e.g. dim=3 vs dim=33);
    shorter ones are zero-padded to the max dimension before fitting and
    unpadded after assignment.

    Args:
        list_of_neuron_tokens: List of INRTokenizer.tokenize outputs,
                               i.e. List[List[List[Tensor]]]
        k: number of centroids / codebook size (default 1000)

    Returns:
        Same nested structure with each neuron replaced by its nearest centroid.
    """
    max_dim = max(
        token.shape[0]
        for neuron_tokens in list_of_neuron_tokens
        for layer in neuron_tokens
        for token in layer
    )
    print(f"Max neuron dimension across all INRs: {max_dim}")

    # Flatten all neurons, zero-padding shorter ones
    all_neurons = []
    all_dims = []
    for neuron_tokens in list_of_neuron_tokens:
        for layer in neuron_tokens:
            for token in layer:
                dim = token.shape[0]
                if dim < max_dim:
                    padded = torch.zeros(max_dim)
                    padded[:dim] = token
                    all_neurons.append(padded)
                else:
                    all_neurons.append(token)
                all_dims.append(dim)
    
    data = torch.stack(all_neurons).numpy()

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels = kmeans.fit_predict(data)
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    # Rebuild the original nested structure, stripping padding where needed
    result = []
    idx = 0
    for neuron_tokens in list_of_neuron_tokens:
        inr_result = []
        for layer in neuron_tokens:
            layer_result = []
            for _ in layer:
                layer_result.append(centroids[labels[idx]][:all_dims[idx]])
                idx += 1
            inr_result.append(layer_result)
        result.append(inr_result)

    return result

def kmeans_discretize(list_of_neuron_tokens: list, k: int = 1000) -> list:
    """
    Quantize neurons from multiple INRs using per-dimension KMeans codebooks.

    Each unique neuron dimension gets its own KMeans fit, avoiding the
    distortion introduced by zero-padding shorter neurons into a shared space.

    Args:
        list_of_neuron_tokens: List of INRTokenizer.tokenize outputs,
                               i.e. List[List[List[Tensor]]]
        k: number of centroids per codebook (capped at the group size)

    Returns:
        Same nested structure with each neuron replaced by its nearest centroid.
    """
    from collections import defaultdict

    # Collect neurons grouped by dimension, tracking their location
    groups = defaultdict(list)  # dim -> [(inr_idx, layer_idx, neuron_idx, tensor)]
    for inr_idx, neuron_tokens in enumerate(list_of_neuron_tokens):
        for layer_idx, layer in enumerate(neuron_tokens):
            for neuron_idx, token in enumerate(layer):
                groups[token.shape[0]].append((inr_idx, layer_idx, neuron_idx, token))

    # Fit a separate KMeans per dimension and store centroid assignments
    assignments = {}  # (inr_idx, layer_idx, neuron_idx) -> centroid tensor
    for dim, entries in groups.items():
        k_eff = min(k, len(entries))
        data = torch.stack([t for _, _, _, t in entries]).numpy()
        print(f"dim={dim}: {len(entries)} neurons, k={k_eff}")

        km = KMeans(n_clusters=k_eff, n_init="auto", random_state=0).fit(data)
        labels = km.labels_
        centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        for i, (inr_idx, layer_idx, neuron_idx, _) in enumerate(entries):
            assignments[(inr_idx, layer_idx, neuron_idx)] = centroids[labels[i]]

    # Rebuild the original nested structure
    return [
        [
            [assignments[(inr_idx, layer_idx, neuron_idx)] for neuron_idx in range(len(layer))]
            for layer_idx, layer in enumerate(neuron_tokens)
        ]
        for inr_idx, neuron_tokens in enumerate(list_of_neuron_tokens)
    ]
