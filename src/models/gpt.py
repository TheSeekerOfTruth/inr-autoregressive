import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # PyTorch 2.0+ Flash Attention execution
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class INRGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Learnable encoders to map raw features up to n_embd 
        self.layer_0_encoder = nn.Linear(3, config.n_embd)
        self.layer_1_encoder = nn.Linear(33, config.n_embd)
        self.layer_2_encoder = nn.Linear(33, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            layer_embeddings = nn.Embedding(config.num_layers, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Learnable decoders to map from hidden states back down to raw dimensions
        self.layer_0_decoder = nn.Linear(config.n_embd, 3)
        self.layer_1_decoder = nn.Linear(config.n_embd, 33)
        self.layer_2_decoder = nn.Linear(config.n_embd, 33)

    def forward(self, flat_x, layer_ids):
        # flat_x shape: (B * T, 33) 
        # layer_ids shape: (B, T)
        B, T = layer_ids.size()
        
        # 1. Flatten layer IDs to match the length of flat_x -> Shape: (B * T)
        flat_layers = layer_ids.view(-1) 
        
        # 2. Pre-allocate an empty uniform grid directly on the GPU
        # This will hold our final unified 384-dimensional embeddings
        neuron_emb = torch.zeros(B * T, self.config.n_embd, device=flat_x.device)
        
        # 3. Create the Boolean Masks
        # mask_0 becomes a tensor of True/False values where layer_id == 0
        mask_0 = (flat_layers == 0)
        mask_1 = (flat_layers == 1)
        mask_2 = (flat_layers == 2)
        
        # 4. Parallel Slicing Projections (No Loops!)
        if mask_0.any():
            neuron_emb[mask_0] = self.layer_0_encoder(flat_x[mask_0, :3])
            
        if mask_1.any():
            neuron_emb[mask_1] = self.layer_1_encoder(flat_x[mask_1])

        if mask_2.any():
            neuron_emb[mask_1] = self.layer_2_encoder(flat_x[mask_2])    
            
        # 5. Reshape back to standard 3D Transformer Layout
        neuron_emb = neuron_emb.view(B, T, self.config.n_embd) # Shape: (B, T, 384)

        # 6. Apply positional contextual layers and pass to transformer blocks
        pos = torch.arange(0, T, dtype=torch.long, device=flat_x.device)
        pos_emb = self.transformer.wpe(pos)
        pos_layers = self.transformer.layer_embeddings(layer_ids)
        x = self.transformer.drop(neuron_emb + pos_emb + pos_layers)
        
        for block in self.transformer.h:
            x = block(x)
            
        return self.transformer.ln_f(x)