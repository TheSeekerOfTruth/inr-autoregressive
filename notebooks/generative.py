import torch
import torch.nn.functional as F
from data_generation.model import mnistINR 

class INRImageGenerator:
    def __init__(self, device, processor, h=28, w=28, beam_width=4, beam_temperature=3.5, top_k=9, temperature=1.1):
        """
        Stateful helper class to manage autoregressive generation modes ('beam_search', 'sampling', 'argmax')
        and reconstruct continuous model weights directly in-memory for immediate spatial image rendering.
        
        Args:
            device: torch.device target
            processor: Your active INRDataProcessor instance (passed from the main script to reuse its tokenizer/discretizer)
        """
        self.device = device
        self.h = h
        self.w = w
        
        # Generation Hyperparameters
        self.beam_width = beam_width
        self.beam_temperature = beam_temperature
        self.top_k = top_k
        self.temperature = temperature
        
        # 🧠 Pass dependencies from your active data stack to access configuration definitions
        self.discretizer = processor.discretizer
        self.tokenizer = processor.tokenizer
        self.delimiter_id = self.discretizer.n_bins + 1
        
        # Pre-calculate static coordinate grids once at initialization
        y_coords = torch.linspace(0, 1, self.h, device=self.device)
        x_coords = torch.linspace(0, 1, self.w, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1) # [784, 2]
        
        # Underlying target model instance container
        self.inr_renderer = mnistINR().to(self.device)
        self.inr_renderer.eval()

    def generate_images(self, gpt_model, num_samples, batch_loader_sample, mode='sampling'):
        """
        Generates a whole batch of images fully in parallel using 'beam_search', 'sampling', 
        or 'argmax', then decodes them via spatial coordinates.
        Returns a [B, 3, H, W] RGB uint8 tensor.
        """
        gpt_model.eval()
        
        max_gen_length = len(batch_loader_sample['x_tokens'][0])
        prompt_length = 1  
        
        # Target token container
        final_gen_tokens = torch.zeros((num_samples, max_gen_length), dtype=torch.long, device=self.device)
        initial_prompts = batch_loader_sample['x_tokens'][:num_samples, :prompt_length].to(self.device)

        with torch.no_grad():
            # =====================================================================
            # OPTION A: STOCHASTIC VECTORIZED BEAM SEARCH (Speed & Temperature Fix)
            # =====================================================================
            if mode == 'beam_search':
                B = num_samples
                W = self.beam_width
                
                # Pre-allocate a trace tracker to record parent branching histories cleanly
                # This completely removes the expensive tensor-copy rewriting step
                parent_history = torch.zeros((B * W, max_gen_length), dtype=torch.long, device=self.device)
                beams = torch.zeros((B * W, max_gen_length), dtype=torch.long, device=self.device)
                
                # Fill step 0
                beams[:, :prompt_length] = initial_prompts.unsqueeze(1).repeat(1, W, 1).view(B * W, prompt_length)
                
                beam_scores = torch.zeros(B, W, device=self.device)
                beam_scores[:, 1:] = float('-inf')  # Force start on path index 0
                beam_scores = beam_scores.view(B * W)
                
                for t in range(prompt_length, max_gen_length):
                    current_context = beams[:, :t]
                    logits, _ = gpt_model(current_context)
                    next_token_log_probs = F.log_softmax(logits[:, -1, :], dim=-1) # [B * W, Vocab]
                    vocab_size = next_token_log_probs.shape[-1]
                    
                    next_token_log_probs = next_token_log_probs.view(B, W, vocab_size)
                    candidate_scores = beam_scores.view(B, W, 1) + next_token_log_probs # [B, W, Vocab]
                    flat_candidate_scores = candidate_scores.view(B, W * vocab_size) # [B, W * Vocab]
                    
                    # 🚀 BEAM TEMPERATURE ACTIVATION: Soften distribution via scaling factor
                    scaled_scores = flat_candidate_scores / self.beam_temperature
                    path_probabilities = F.softmax(scaled_scores, dim=-1)
                    
                    # Sample 2*W potential pathways to handle soft/diverse choices safely
                    sampled_indices = torch.multinomial(path_probabilities, num_samples=min(2 * W, vocab_size * W), replacement=False)
                    sampled_scores = torch.gather(flat_candidate_scores, 1, sampled_indices)
                    
                    # Filter and sort back down to the target Beam Width (W)
                    sorted_lift = torch.argsort(sampled_scores, descending=True, dim=-1)
                    top_indices = torch.gather(sampled_indices, 1, sorted_lift)[:, :W]
                    top_scores = torch.gather(flat_candidate_scores, 1, top_indices)
                    
                    parent_beam_ids = torch.div(top_indices, vocab_size, rounding_mode='floor')
                    token_ids = top_indices % vocab_size
                    
                    batch_offsets = torch.arange(B, device=self.device).unsqueeze(1) * W
                    flat_parent_idx = (parent_beam_ids + batch_offsets).view(-1)
                    
                    # Append new choices and write tracks natively via optimized indexing
                    beams[:, :t] = beams[flat_parent_idx, :t]
                    beams[:, t] = token_ids.view(-1)
                    
                    parent_history[:, t] = parent_beam_ids.view(-1)
                    beam_scores = top_scores.view(B * W)
                
                # Backtrack out the single winning sequence pathway per batch item
                final_scores = beam_scores.view(B, W)
                best_beam_indices = torch.argmax(final_scores, dim=-1)
                
                final_batch_offsets = torch.arange(B, device=self.device) * W
                best_flat_indices = best_beam_indices + final_batch_offsets
                
                final_gen_tokens = beams[best_flat_indices]

            # =====================================================================
            # OPTION B: TOP-K SAMPLING (Fully Batched)
            # =====================================================================
            elif mode == 'sampling':
                final_gen_tokens[:, :prompt_length] = initial_prompts
                for t in range(prompt_length, max_gen_length):
                    current_context = final_gen_tokens[:, :t]
                    logits, _ = gpt_model(current_context)
                    next_token_logits = logits[:, -1, :] 
                    
                    if self.temperature != 1.0:
                        next_token_logits = next_token_logits / self.temperature
                        
                    topk_values, _ = torch.topk(next_token_logits, self.top_k, dim=-1)
                    min_values = topk_values[:, -1].unsqueeze(-1)
                    
                    masked_logits = torch.where(
                        next_token_logits >= min_values,
                        next_token_logits,
                        torch.full_like(next_token_logits, float('-inf'))
                    )
                    
                    probs = F.softmax(masked_logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1)
                    final_gen_tokens[:, t] = next_token_ids.squeeze(-1)

            # =====================================================================
            # OPTION C: GREEDY ARGMAX (Fully Batched)
            # =====================================================================
            else:
                final_gen_tokens[:, :prompt_length] = initial_prompts
                for t in range(prompt_length, max_gen_length):
                    current_context = final_gen_tokens[:, :t]
                    logits, _ = gpt_model(current_context)
                    next_token_logits = logits[:, -1, :]
                    next_token_ids = torch.argmax(next_token_logits, dim=-1)
                    final_gen_tokens[:, t] = next_token_ids

        # =====================================================================
        # 🚀 INLINED IN-MEMORY WEIGHT RECONSTRUCTION & RENDERING
        # =====================================================================
        processed_images = []
        layer_ids = batch_loader_sample['layer_ids'][0].cpu()
        unique_layers = torch.unique(layer_ids).tolist()
        target_length = len(layer_ids) # Exact structure expected: 153

        for idx in range(num_samples):
            # Strip prompt start index to match tokens[1:] behavior
            tokens = final_gen_tokens[idx].cpu()[1:]
            
            # --- 1. IDENTIFY AND STRIP DELIMITERS ---
            valid_token_mask = (tokens != self.delimiter_id)
            clean_tokens = tokens[valid_token_mask]
            
            # --- 🛡️ CRASH GUARD: Force sequence length to match 153 no matter what ---
            clean_tokens = clean_tokens[:target_length]
            if len(clean_tokens) < target_length:
                padding = torch.zeros(target_length - len(clean_tokens), dtype=clean_tokens.dtype)
                clean_tokens = torch.cat([clean_tokens, padding], dim=0)
            
            # --- 2. DECODE FILTERED CONTINUOUS VALUES ---
            decoded_weights = self.discretizer.decode(clean_tokens)
            
            # --- 3. RECONSTRUCT MODEL LAYERS ---
            layers_reconstructed = []
            for l_id in unique_layers:
                mask = (layer_ids == l_id).view(-1)
                layer_matrix = decoded_weights[mask]
                
                if self.tokenizer.token == "neuron":
                    layer_neurons = list(torch.unbind(layer_matrix, dim=0))         
                else:
                    layer_neurons = [layer_matrix]         
                    
                layers_reconstructed.append(layer_neurons)
                
            # Get the exact model state_dict mapping purely in RAM
            state_dict = self.tokenizer.detokenize(layers_reconstructed)

            # Load weights into renderer and project spatial coordinates
            try:
                self.inr_renderer.load_state_dict(state_dict)
            except Exception:
                continue
                
            with torch.no_grad():
                reconstruction = self.inr_renderer(self.coords).view(1, self.h, self.w)
                
                recon_min, recon_max = reconstruction.min(), reconstruction.max()
                if recon_max - recon_min > 1e-5:
                    reconstruction = (reconstruction - recon_min) / (recon_max - recon_min)
                
                img_uint8 = (reconstruction * 255.0).clamp(0, 255).to(torch.uint8)
                img_rgb = img_uint8.repeat(3, 1, 1) 
                processed_images.append(img_rgb)

        if len(processed_images) == 0:
            return torch.zeros((1, 3, self.h, self.w), dtype=torch.uint8, device=self.device)
            
        return torch.stack(processed_images).to(self.device)