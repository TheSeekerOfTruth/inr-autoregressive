import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Union
from src.utils.tokenizer import INRTokenizer
from src.utils.discretizer import INRDiscretizer

class INRDataProcessor:   
    def __init__(
        self, 
        input_root: str = "data/mnist-inrs-relus/all_inrs_eight_neurons", 
        output_root: str = "data/processed_inrs", 
        n_bins: int = 150,
        mode: str = "quantile",
        token: str = "weight"
    ):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.tokenizer = INRTokenizer(token = token)
        self.discretizer = INRDiscretizer(mode=mode, n_bins=n_bins, input_path=Path(input_root))

    def _process_single_file(self, file_path: Path, destination: Path):
        """Standard Forward: .pth -> .pt (flattened & digitized)"""        
        state_dict = torch.load(file_path, map_location='cpu')
        tokenized_layers = self.tokenizer.tokenize(state_dict)
        discretized_layers = self.discretizer.discretize(tokenized_layers)
        delimiter = self.discretizer.n_bins + 1

        if(self.tokenizer.token == "neuron"):
            flattened_layers = [token for layer in discretized_layers for token in layer]
        else:
            all_elements = []
            dim = 9
            if isinstance(discretized_layers, torch.Tensor):
                current_idx = 0
                for layer in tokenized_layers:
                    for token in layer:
                        num_elements = token.numel()
                        quantized_token = discretized_layers[current_idx : current_idx + num_elements]
                        current_idx += num_elements                      
                        neurons = num_elements / dim
                        all_elements.append(torch.cat([
                            quantized_token.view(-1, dim),
                            torch.tensor([[delimiter]], dtype=torch.long).repeat(int(neurons), 1)
                        ], dim=1).flatten())
            else:
                for _, layer in enumerate(discretized_layers):
                    for token in layer:
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

        save_name = file_path.parent.name + "_" + file_path.stem + ".pt"
        if(self.tokenizer.token == "neuron"):
            data = torch.stack(flattened_layers, dim=0)
        else:
            data = flattened_layers
        torch.save({
            'tokens': data, 
            'layer_ids': torch.tensor(layer_ids, dtype=torch.long)
        }, destination / save_name)

    def reconstruct_state_dict(self, processed_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """
        Takes a processed .pt file, strips out autoregressive delimiters, 
        and converts it back into a standard model state_dict.
        """
        data = torch.load(processed_path, map_location='cpu')
        tokens = data['tokens']      
        layer_ids = data['layer_ids']  
        
        delimiter_id = self.discretizer.n_bins + 1
        valid_token_mask = (tokens != delimiter_id)
        
        # Filter tokens down to just the actual weights/biases
        clean_tokens = tokens[valid_token_mask]
        
        # 2. Decode the filtered continuous data
        print("\n" + "="*40)
        print("🔍 DEBUGGING: TOKENS & DECODED WEIGHTS")
        print("="*40)
        
        # 1. Print Tokens Info
        print(f"🔹 Tokens:\n{tokens}")
        print(f"📐 Tokens Shape: {list(tokens.shape)}\n")
        print("-" * 30)
        
        # 2. Decode and Print Weights Info
        decoded_weights = self.discretizer.decode(clean_tokens)
        print(f"🔸 Decoded Weights:\n{decoded_weights}")
        print(f"📐 Decoded Weights Shape: {list(decoded_weights.shape)}")
        
        print("="*40 + "\n")
        unique_layers = torch.unique(layer_ids).tolist()
        layers_reconstructed = []
        
        # 3. Slice up the layer matrices matching layer_ids exactly
        for l_id in unique_layers:
            mask = (layer_ids == l_id).view(-1)
            layer_matrix = decoded_weights[mask]
            
            if self.tokenizer.token == "neuron":
                layer_neurons = list(torch.unbind(layer_matrix, dim=0))         
            else:
                layer_neurons = [layer_matrix]         
                
            layers_reconstructed.append(layer_neurons)
            
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
        src = self.input_root 
        dst = self.output_root
        dst.mkdir(parents=True, exist_ok=True)
        files = list(src.rglob("*.pth"))
        for f in tqdm(files, desc="Processing files"):
            self._process_single_file(f, dst)

if __name__ == "__main__":     
    processor = INRDataProcessor()
    processor.run()