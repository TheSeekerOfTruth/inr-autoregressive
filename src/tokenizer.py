import torch
import torch.nn.functional as F
from collections import OrderedDict

class INRTokenizer:
    def __init__(self, token = "neuron", sort = True):
        self.token = token
        self.sort = sort

    def _canonicalize(self, state_dict):
            """
            Sorts the hidden neurons functionally to eliminate permutation invariance 
            WITHOUT destroying the mathematical function of the original MLP.
            """
            # Create a deep copy to keep the original state_dict safe
            aligned = {k: v.clone() for k, v in state_dict.items()}
            
            # 1. Canonicalize between Layer 0 and Layer 1
            w0 = aligned['seq.0.weight']  # Shape: (32, 2)
            b0 = aligned['seq.0.bias']    # Shape: (32,)
            w1 = aligned['seq.1.weight']  # Shape: (32, 32)
            
            # Sort Layer 0 neurons by their row magnitude (L2-norm)
            norm0 = torch.norm(w0, p=2, dim=1)
            indices0 = torch.argsort(norm0)
            
            aligned['seq.0.weight'] = w0[indices0]
            aligned['seq.0.bias'] = b0[indices0]
            # Crucial: Permute the inputs (columns) of Layer 1 to match!
            aligned['seq.1.weight'] = w1[:, indices0]

            # 2. Canonicalize between Layer 1 and Layer 2
            w1_aligned = aligned['seq.1.weight'] # Shape: (32, 32)
            b1 = aligned['seq.1.bias']           # Shape: (32,)
            w2 = aligned['seq.2.weight']         # Shape: (1, 32)
            
            # Sort Layer 1 neurons by their row magnitude
            norm1 = torch.norm(w1_aligned, p=2, dim=1)
            indices1 = torch.argsort(norm1)
            
            aligned['seq.1.weight'] = w1_aligned[indices1]
            aligned['seq.1.bias'] = b1[indices1]
            # Crucial: Permute the inputs (columns) of Layer 2 to match!
            aligned['seq.2.weight'] = w2[:, indices1]
            
            return aligned
    
    def tokenize(self, state_dict):
        """
        ENCODE: state_dict(MLP model) -> tokenized_layers
        """
        if(self.sort):
            state_dict = self._canonicalize(state_dict)
        neuron_tokens = []

        # Identify layers in the model
        keys = state_dict.keys()
        layer_indices = sorted(list(set(int(k.split('.')[1]) for k in keys if 'seq' in k)))

        for idx in layer_indices:
            w = state_dict[f'seq.{idx}.weight']
            b = state_dict[f'seq.{idx}.bias']
            if(self.token == "neuron"):    
                """ F.pad(x, (0, 33 - x.size(-1))) """       
                layer_tokens = [x for x in torch.cat([w, b.unsqueeze(1)], dim=1)]
                neuron_tokens.append(layer_tokens)
            elif(self.token == "weight"):
                flat_layer = torch.cat([w, b.unsqueeze(1)], dim=1).flatten()
                neuron_tokens.append([flat_layer])
                
        # This is a List containting Lists (each List represent a Layer having all its tokens as tensors)
        return neuron_tokens

    def detokenize(self, list_of_layers):
        """
        DECODE: tokenized_layers -> state_dict
        """
        state_dict = OrderedDict()
        
        for i, layer_tokens in enumerate(list_of_layers):
            
            if(self.token == "neuron"):
                if(i == 0):
                    layer_as_tensor = torch.stack(layer_tokens, dim=0).reshape(32, 33) 
                    w = layer_as_tensor[:, :2]
                    b = layer_as_tensor[:, 2]
                elif(i == 1):
                    layer_as_tensor = torch.stack(layer_tokens, dim=0).reshape(32, 33) 
                    w = layer_as_tensor[:, :-1]
                    b = layer_as_tensor[:, -1]
                else:
                    layer_as_tensor = torch.stack(layer_tokens, dim=0).reshape(1, 33) 
                    w = layer_as_tensor[:, :-1]
                    b = layer_as_tensor[:, -1]
            
            else:
                big_layer_tensor = layer_tokens[0]
                
                if(i == 0):
                    layer_as_tensor = big_layer_tensor.reshape(32, 3)
                    w = layer_as_tensor[:, :2]
                    b = layer_as_tensor[:, 2]
                elif(i == 1):
                    layer_as_tensor = big_layer_tensor.reshape(32, 33)
                    w = layer_as_tensor[:, :-1]
                    b = layer_as_tensor[:, -1]
                else:
                    layer_as_tensor = big_layer_tensor.reshape(1, 33)
                    w = layer_as_tensor[:, :-1]
                    b = layer_as_tensor[:, -1]
            
            state_dict[f'seq.{i}.weight'] = w
            state_dict[f'seq.{i}.bias'] = b
            
        return state_dict
    