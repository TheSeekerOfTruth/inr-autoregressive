import torch
import numpy as np

class INRDiscretizer:
    def __init__(self, mode='quantile', n_bins=50, k=50, min_val=-5, max_val=5, input_path=None):
        self.mode = mode
        self.n_bins = n_bins
        self.k = k
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val
        self.bin_edges = None
        self.bin_centers = None
        self.input_root = input_path
        if self.input_root is not None and self.mode == 'quantile':
            sample_weights = []            
            if self.input_root is not None:
                pth_files = list(self.input_root.rglob("*.pth"))             
                for file in pth_files[:50]:
                    try:
                        sd = torch.load(file, map_location='cpu')
                        for k_key, v in sd.items():
                            if 'weight' in k_key or 'bias' in k_key:
                                sample_weights.append(v.flatten())
                    except Exception:
                        continue            
            if len(sample_weights) == 0:
                raise ValueError(
                    f"Could not find any .pth files inside '{self.input_root}'. "
                    f"Verify that your path correctly steps out of your execution folder if running a notebook."
                )                
            all_raw_weights = torch.cat(sample_weights, dim=0)
            self._compute_quantile_bins(all_raw_weights)
    
    def _compute_quantile_bins(self, weights):
        """Calculates bin thresholds based entirely on data density."""
        if isinstance(weights, torch.Tensor):
            weights_np = weights.detach().cpu().float().numpy().flatten()
        else:
            weights_np = np.asarray(weights).flatten()
            
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        edges = np.percentile(weights_np, quantiles)
        edges[0] = max(edges[0], self.min_val)
        edges[-1] = min(edges[-1], self.max_val)
        centers = []
        for i in range(self.n_bins):
            mask = (weights_np >= edges[i]) & (weights_np <= edges[i+1])
            if np.any(mask):
                centers.append(np.median(weights_np[mask]))
            else:
                centers.append((edges[i] + edges[i+1]) / 2.0)                
        self.bin_edges = torch.tensor(edges, dtype=torch.float32)
        self.bin_centers = torch.tensor(centers, dtype=torch.float32)
        
        print(f"Quantile Bins initialized over {self.n_bins} spans.")
        print(f" -> High-Density Center Bin Resolution Width: {edges[self.n_bins//2 + 1] - edges[self.n_bins//2]:.5f}")
        print(f" -> Low-Density Tail Bin Resolution Width: {edges[1] - edges[0]:.5f}")

    def discretize(self, list_of_neuron_tokens: list) -> list:
        """Entry point for all discretization modes."""
        if self.mode == 'uniform':
            return [[self._quantize_uniform(t) for t in layer] for layer in list_of_neuron_tokens]
        elif self.mode == 'quantile':
            flat_scalars = []
            for layer in list_of_neuron_tokens:
                for t in layer:
                    if isinstance(t, torch.Tensor):
                        if t.dim() == 0:
                            flat_scalars.append(t.item())
                        else:
                            for val in t.flatten():
                                flat_scalars.append(val.item())
                    else:
                        flat_scalars.append(float(t))  
            all_weights = torch.tensor(flat_scalars, dtype=torch.float32)
            bin_indices = self._quantize_quantile(all_weights)
            unique_tokens = bin_indices            
            return unique_tokens


    def _quantize_uniform(self, x: torch.Tensor) -> torch.Tensor:        
        normalized = (x - self.min_val) / self.range
        idx = (normalized * (self.n_bins - 1)).round().clamp(0, self.n_bins - 1)
        return idx.long()

    def _quantize_quantile(self, x: torch.Tensor) -> torch.Tensor:
        """Maps continuous values to dynamic density-optimized bins using bucket search."""
        idx = torch.bucketize(x, self.bin_edges) - 1
        return idx.clamp(0, self.n_bins - 1).long()


    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Maps global tokenized indices back to their optimal continuous coordinates."""
        if self.mode == "uniform":
            valid_mask = (indices != (self.n_bins + 1))
            output = torch.zeros_like(indices, dtype=torch.float32)
            output[valid_mask] = self.get_vocabulary()[indices[valid_mask]]
            return output
        elif self.mode == "quantile":
            valid_mask = (indices != (self.n_bins + 1))
            output = torch.zeros_like(valid_mask, dtype=torch.float32)
            output[valid_mask] = self.bin_centers[indices[valid_mask].long()]
            return output

    def get_vocabulary(self) -> torch.Tensor:
        """Returns the fallback uniform codebook."""
        steps = torch.arange(0, self.n_bins, dtype=torch.float32)
        return (steps / (self.n_bins - 1)) * self.range + self.min_val

