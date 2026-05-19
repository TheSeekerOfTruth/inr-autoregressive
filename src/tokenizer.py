import torch
import torch.nn.functional as F
from collections import OrderedDict

class INRTokenizer:
    def __init__(self):
        pass

    def tokenize(self, state_dict):
        """
        ENCODE: state_dict(MLP model) -> tokenized_layers
        """
        neuron_tokens = []

        # Identify layers in the model
        keys = state_dict.keys()
        layer_indices = sorted(list(set(int(k.split('.')[1]) for k in keys if 'seq' in k)))

        for idx in layer_indices:
            w = state_dict[f'seq.{idx}.weight']
            b = state_dict[f'seq.{idx}.bias']           
            layer_tokens = [F.pad(x, (0, 33 - x.size(-1))) for x in torch.cat([w, b.unsqueeze(1)], dim=1)]
            neuron_tokens.append(layer_tokens)
        
        # This is a List containting Lists (each List represent a Layer having all its tokens/neurons as tensors)
        return neuron_tokens

    def detokenize(self, list_of_layers):
        """
        DECODE: tokenized_layers -> state_dict
        """
        state_dict = OrderedDict()
        
        for i, layer_tokens in enumerate(list_of_layers):

            layer_as_tensor = torch.stack(layer_tokens, dim=0) 
            if(i == 0):
                w = layer_as_tensor[:, :2]
                b = layer_as_tensor[:, 2]
            else:
                w = layer_as_tensor[:, :-1]
                b = layer_as_tensor[:, -1]
            
            state_dict[f'seq.{i}.weight'] = w
            state_dict[f'seq.{i}.bias'] = b
            
        return state_dict