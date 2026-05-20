import torch
import torch.nn.functional as F
from collections import OrderedDict

class INRTokenizer:
    def __init__(self, token = "neuron"):
        self.token = token

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
            if(self.token == "neuron"):           
                layer_tokens = [F.pad(x, (0, 33 - x.size(-1))) for x in torch.cat([w, b.unsqueeze(1)], dim=1)]
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
                    layer_as_tensor = torch.stack(layer_tokens, dim=0).reshape(32, 3) 
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