import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Union
from src.tokenizer import INRTokenizer
from src.discretizer import INRDiscretizer
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np

def fit(list_of_neuron_tokens: list, save_path: str = "quantizer_data.pt", k: int = 50):
    groups = defaultdict(list)
    codebooks = {}
    
    # 1. Verbose Loading
    print("--- Loading and grouping tokens ---")
    for neuron_tokens in tqdm(list_of_neuron_tokens, desc="Grouping tokens by dimension"):
        for layer in neuron_tokens:
            for token in layer:
                groups[token.shape[0]].append(token.numpy())

    # 2. Verbose Training
    print("--- Starting KMeans fitting ---")
    for dim, entries in groups.items():
        print(f"Fitting KMeans for dimension {dim} (Total samples: {len(entries)})...")
        
        k_eff = min(k, len(entries))
        data = np.stack(entries)
        
        # This is where your code usually hangs/crashes
        km = KMeans(n_clusters=k_eff, n_init="auto", random_state=0).fit(data)
        
        codebooks[dim] = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        print(f"Finished fitting dimension {dim}.")

    torch.save({'codebooks': codebooks}, save_path)
    print(f"Quantizer data saved to {save_path}")

class INRDataProcessor:
    
    def __init__(
        self, 
        input_root: str = "data/mnist-inrs", 
        output_root: str = "data/processed_inrs", 
        n_bins: int = 50,
        mode: str = "kmeans",
        codebook_path: Path = "quantizer_data.pt"
    ):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.tokenizer = INRTokenizer(token = "neuron")
        self.discretizer = INRDiscretizer(mode=mode, n_bins=n_bins,  codebooks=torch.load(codebook_path)['codebooks'])

    def _process_single_file(self, file_path: Path, destination: Path):
        """Standard Forward: .pth -> .pt (flattened & digitized)"""
        try:
            state_dict = torch.load(file_path, map_location='cpu')
            tokenized_layers = self.tokenizer.tokenize(state_dict)
            discretized_layers = self.discretizer.discretize(tokenized_layers)
            delimiter = self.discretizer.n_bins + 1


            if(self.tokenizer.token == "neuron"):
                flattened_layers = [token for layer in discretized_layers for token in layer]
            else:
                all_elements = []

                # 1. Unpack the outer list (layers)
                for idx, layer in enumerate(discretized_layers):
                    # 2. Unpack the inner list (neurons)
                    for token in layer:
                        if(idx == 0):
                            dim = 3
                        else:
                            dim = 33
                        neurons = token.shape[0] / dim
                        all_elements.append(torch.cat([token.view(-1,dim),torch.tensor([[delimiter]], dtype=torch.long).repeat(int(neurons),1)], dim=1).flatten())

                flattened_layers = torch.cat(all_elements, dim=0)

            layer_ids = []
            for l_idx, layer in enumerate(tokenized_layers):
                if self.tokenizer.token == "neuron":
                    element_count = len(layer)
                else:
                    element_count = layer[0].numel()
                layer_ids.extend([l_idx] * element_count)

            save_name = file_path.parent.parent.name + ".pt"
            if(self.tokenizer.token == "neuron"):
                data = torch.stack(flattened_layers, dim=0)
            else:
                data = flattened_layers
            torch.save({
                'tokens': data, 
                'layer_ids': torch.tensor(layer_ids, dtype=torch.long)
            }, destination / save_name)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    def reconstruct_state_dict(self, processed_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """
        Takes a processed .pt file and converts it back into a 
        standard model state_dict (weights and biases).
        """
        data = torch.load(processed_path, map_location='cpu')
        tokens = data['tokens']
        print(tokens)
        layer_ids = data['layer_ids']  
        unique_layers = torch.unique(layer_ids).tolist()
        layers_reconstructed = []
        
        for l_id in unique_layers:
            mask = (layer_ids == l_id).view(-1)
            flat_tokens = tokens.view(-1)
            layer_matrix = flat_tokens[mask]
            if self.tokenizer.token == "neuron":
                layer_neurons = list(torch.unbind(layer_matrix, dim=0))         
            else:
                layer_neurons = [layer_matrix]         
            layers_reconstructed.append(layer_neurons)
            print(layers_reconstructed)
        state_dict = self.tokenizer.detokenize(layers_reconstructed)       
        return state_dict

    def save_reconstructed_pth(self, processed_path: str, output_path: str):
        """
        Helper to convert a GPT-ready .pt file back into a .pth checkpoint 
        ready for the BatchSiren model.
        """
        state_dict = self.reconstruct_state_dict(processed_path)
        torch.save(state_dict, output_path)
        print(f"Successfully reconstructed checkpoint to: {output_path}")


    def run(self):
        """Process all train/test files."""
        for split in ["training", "testing"]:
            src = self.input_root / split
            dst = self.output_root / split
            if not src.exists(): continue
            dst.mkdir(parents=True, exist_ok=True)
            files = list(src.rglob("*.pth"))
            for f in tqdm(files, desc=f"Processing {split}"):
                self._process_single_file(f, dst)

if __name__ == "__main__":     
    list_for_kmeans = []
    tokenizer = INRTokenizer(token = "neuron")   
    for file in tqdm(list(Path("data/mnist-inrs/training").rglob("*.pth"))):
        state_dict = torch.load(file, map_location='cpu')
        list_for_kmeans.append(tokenizer.tokenize(state_dict))
    fit(list_for_kmeans)
    processor = INRDataProcessor()
    processor.run()