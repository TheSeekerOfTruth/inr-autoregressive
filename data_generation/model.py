import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L

    def forward(self, x):
        out = []
        for i in range(self.L):
            freq = 2.0 ** i
            out.append(torch.sin(freq * np.pi * x))
            out.append(torch.cos(freq * np.pi * x))
        return torch.cat(out, dim=-1)
    
class mnistINR(nn.Module):
    def __init__(self, nb_layers=2, nb_neurons=8, L=2):
        super().__init__()
        self.pos_encoder = PositionalEncoding(L=L)

        # Each coordinate gets mapped to L sin waves and L cos waves -> 4*L is the input_dim
        input_dim = 4 * L 
        curr_dim = input_dim

        layers = []

        for _ in range(nb_layers):
            layers.append(nn.Linear(curr_dim, nb_neurons))
            layers.append(nn.ReLU())
            curr_dim = nb_neurons
        
        layers.append(nn.Linear(curr_dim, 1))
        layers.append(nn.Sigmoid())
        self.seq = nn.Sequential(*layers)
    
    def forward(self, coords):
        return self.seq(self.pos_encoder(coords))
