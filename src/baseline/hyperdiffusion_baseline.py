import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision

# =====================================================================
# 1. LIVE DIRECTORY CONCATENATION READER
# =====================================================================
def load_flat_inr_directory(root_dir, target_dim):
    all_flat_networks = []
    print(f"Scanning directory path: '{root_dir}'...")
    if not os.path.exists(root_dir):
        print(f"⚠️ Warning: Directory '{root_dir}' not found.")
        return None
        
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pth") or file.endswith(".pt"):
                file_path = os.path.join(root, file)
                try:
                    data = torch.load(file_path, map_location="cpu")
                    if isinstance(data, dict):
                        flat_weights = torch.cat([p.flatten() for p in data.values()])
                    elif isinstance(data, torch.Tensor):
                        flat_weights = data.flatten()
                    else:
                        continue
                    if flat_weights.shape[0] == target_dim:
                        all_flat_networks.append(flat_weights)
                except Exception as e:
                    pass
    if len(all_flat_networks) == 0:
        return None
    return torch.stack(all_flat_networks)


# =====================================================================
# 2. THE EXACT HYPERDIFFUSION BACKBONE
# =====================================================================
class LayerWiseInputProjection(nn.Module):
    def __init__(self, inr_shapes, n_embd):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(math.prod(shape), n_embd) for shape in inr_shapes])
    def forward(self, x_list):
        return torch.stack([proj(x) for proj, x in zip(self.projs, x_list)], dim=1)

class LayerWiseOutputProjection(nn.Module):
    def __init__(self, inr_shapes, n_embd):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(n_embd, math.prod(shape)) for shape in inr_shapes])
    def forward(self, transformer_outputs):
        return torch.cat([proj(transformer_outputs[:, i, :]) for i, proj in enumerate(self.projs)], dim=1)

class HyperDiffusionTransformer(nn.Module):
    def __init__(self, total_dim, n_embd=256, n_layer=12, n_head=8):
        super().__init__()
        self.total_dim = total_dim
        # Map each individual parameter (token) to embedding space
        self.input_proj = nn.Linear(1, n_embd) 
        # Map back from embedding space to individual parameter values
        self.output_proj = nn.Linear(n_embd, 1)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4), nn.SiLU(), nn.Linear(n_embd * 4, n_embd)
        )
        # Positional embedding for 153 individual tokens
        self.pos_emb = nn.Parameter(torch.randn(1, total_dim, n_embd))
        
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=n_embd * 4,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layer)

    def get_timestep_embedding(self, t, dim):
        half = dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    def forward(self, x_flat, t):
        # x_flat shape: [batch, 153]
        # Reshape to [batch, 153, 1]
        h = self.input_proj(x_flat.unsqueeze(-1)) # [batch, 153, n_embd]
        
        # CALL THE HELPER FUNCTION HERE
        t_emb = self.time_mlp(self.get_timestep_embedding(t, h.shape[-1])).unsqueeze(1)
        
        # Add positional embedding and time embedding
        # Ensure pos_emb matches shape: [1, 153, n_embd]
        h = self.transformer(h + self.pos_emb + t_emb)
        
        # Project back to [batch, 153, 1] then flatten
        return self.output_proj(h).squeeze(-1)

# =====================================================================
# 3. DYNAMIC RECONSTRUCTION HELPERS (1D Array -> Functional INR MLP)
# =====================================================================
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
    
class TargetMNISTINR(nn.Module):
    def __init__(self, layer_shapes, L=2):
        super().__init__()
        self.pos_encoder = PositionalEncoding(L=L)
        
        # Calculate structure dynamically from the layout list
        # We assume your layout provides pairs of (weight_shape, bias_shape)
        # Your layout: [(8, 8), (8,), (8, 8), (8,), (1, 8), (1,)]
        
        layers = []
        # We iterate by 2 because each layer has 1 Weight and 1 Bias
        for i in range(0, len(layer_shapes), 2):
            w_shape = layer_shapes[i]
            # w_shape[1] is input dim, w_shape[0] is output dim
            linear = nn.Linear(w_shape[1], w_shape[0])
            layers.append(linear)
            
            # Add activation based on if it's the last layer
            if i < len(layer_shapes) - 2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
        
        self.seq = nn.Sequential(*layers)
    
    def forward(self, coords):
        return self.seq(self.pos_encoder(coords))


def deserialize_weights_to_inr(flat_vector, layer_shapes):
    model = TargetMNISTINR(layer_shapes)
    state_dict = {}
    curr = 0
    
    # We only want to target the indices where nn.Linear layers exist
    # In your TargetMNISTINR(seq), Linear layers are at 0, 2, 4
    linear_indices = [0, 2, 4] 
    
    for idx, i in enumerate(range(0, len(layer_shapes), 2)):
        w_shape = layer_shapes[i]
        b_shape = layer_shapes[i+1]
        
        w_size = math.prod(w_shape)
        b_size = math.prod(b_shape)
        
        w_tensor = flat_vector[curr : curr + w_size].view(w_shape)
        curr += w_size
        b_tensor = flat_vector[curr : curr + b_size].view(b_shape)
        curr += b_size
        
        # Use the 'seq' naming convention
        seq_idx = linear_indices[idx]
        state_dict[f"seq.{seq_idx}.weight"] = w_tensor
        state_dict[f"seq.{seq_idx}.bias"] = b_tensor
        
    model.load_state_dict(state_dict)
    return model


# =====================================================================
# 4. TRACKING METRICS (LeNet Classifier & Score Engine)
# =====================================================================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Feature Extractor
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1), # Output: 24x24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                             # Output: 12x12
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), # Output: 8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                              # Output: 4x4
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 4 * 4, out_features=120),               # Changed from 16*5*5 to 16*4*4
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) 
        logits = self.classifier(x)
        return logits

def calculate_lenet_score(rendered_images, classifier, splits=1):
    classifier.eval()
    
    # Core Data Type & Scale Adjustments (Ensuring we are floats scaled [0,1])
    images = rendered_images.float()
    if images.max() > 1.0:
        images = images / 255.0
        
    if images.shape[1] != 1:
        images = images.mean(dim=1, keepdim=True)

    with torch.no_grad():
        logits = classifier(images)
        preds = F.softmax(logits, dim=1).cpu().numpy()
    
    scores = []
    num_samples = preds.shape[0]
    split_size = num_samples // splits
    
    for i in range(splits):
        part = preds[i * split_size : (i + 1) * split_size, :]
        p_y = np.expand_dims(np.mean(part, axis=0), 0)
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return np.mean(scores)


# =====================================================================
# 5. ALL-IN-ONE PIPELINE TRAINER & VALIDATOR
# =====================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints/generated_inrs", exist_ok=True)
    
    # Initialize TensorBoard Log Writer
    writer = SummaryWriter(log_dir="runs/hyperdiffusion_experiment")
    print("📈 TensorBoard SummaryWriter successfully mounted to 'runs/'")
    
    # Layout shapes alternating weights and biases for TargetMNISTINR
    mnist_layout = [(8, 8), (8,), (8, 8), (8,), (1, 8), (1,)]
    total_dim = 153
    print(total_dim)
    model = HyperDiffusionTransformer(total_dim=153, n_embd=200, n_layer=6, n_head=4).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Mount Pretrained Metrics Engine
    lenet_evaluator = LeNet5().to(device)
    LENET_CHECKPOINT_PATH = "../../MNIST-classifier/lenet5_mnist.pth"
    if os.path.exists(LENET_CHECKPOINT_PATH):
        lenet_evaluator.load_state_dict(torch.load(LENET_CHECKPOINT_PATH, map_location=device)['model_state_dict'])
        print("✅ Pretrained LeNet performance evaluator attached successfully.")
    else:
        print("⚠️ Warning: Pretrained LeNet weights missing. Evaluating using raw initialization weights.")
    lenet_evaluator.eval()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    num_steps = 1000
    betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    sqrt_alphas = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]]) ) / (1.0 - alphas_cumprod)

    # --- SOLIDIFIED TRAIN/VALIDATION SPLIT MANAGEMENT ---
    raw_loaded_data = load_flat_inr_directory("../../data/checkpoints-rec/all_inrs_eight_neurons", total_dim)
    
    if raw_loaded_data is not None:
        print(f"Successfully scraped database. Total records extracted: {raw_loaded_data.size(0)}")
        # Split data deterministically: 90% Train, 10% Validation
        num_samples = raw_loaded_data.size(0)
        split_idx = int(num_samples * 0.9)
        
        # Shuffle indices once deterministically before split
        shuffled_indices = torch.randperm(num_samples)
        train_indices = shuffled_indices[:split_idx]
        val_indices = shuffled_indices[split_idx:]
        print(len(train_indices))
        print(len(val_indices))
        train_matrix = raw_loaded_data[train_indices].to(device)
        #train_matrix = train_matrix[:1000]
        val_matrix = raw_loaded_data[val_indices].to(device)
        #val_matrix = val_matrix[:100]
        print(f"📊 Dataset Split Allocated: {train_matrix.size(0)} Training | {val_matrix.size(0)} Validation.")
    else:
        print("Fallback: Creating random mock variables for local directory tracking emulation.")
        train_matrix = torch.randn(100, total_dim, device=device)
        val_matrix = torch.randn(20, total_dim, device=device)


    batch_size = 512
    print("\nBeginning validation-tracked baseline execution loop...")
    
    global_step = 0
    for epoch in range(100): # Extended epoch span to trace trends cleanly on Tensorboard
        # ──────────────── TRAINING PARTITION PASS ────────────────
        model.train()
        perm = torch.randperm(train_matrix.size(0))
        epoch_train_loss = 0.0
        
        for i in range(0, train_matrix.size(0), batch_size):
            indices = perm[i : i + batch_size]
            clean_batch = train_matrix[indices]
            b_sz = clean_batch.shape[0]
            
            optimizer.zero_grad()
            t = torch.randint(0, num_steps, (b_sz,), device=device).long()
            noise = torch.randn_like(clean_batch)
            
            noisy_batch = sqrt_alphas[t].unsqueeze(1) * clean_batch + sqrt_one_minus_alphas[t].unsqueeze(1) * noise
            loss = F.mse_loss(model(noisy_batch, t), noise)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item() * b_sz
            
            # Log continuous mini-batch updates to Tensorboard
            writer.add_scalar("Loss/Minibatch-Train", loss.item(), global_step)
            global_step += 1

        avg_train_loss = epoch_train_loss / train_matrix.size(0)

        # ──────────────── VALIDATION LOSS EVALUATION ────────────────
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for i in range(0, val_matrix.size(0), batch_size):
                v_batch = val_matrix[i : i + batch_size]
                b_sz = v_batch.shape[0]
                t = torch.randint(0, num_steps, (b_sz,), device=device).long()
                noise = torch.randn_like(v_batch)
                
                noisy_v_batch = sqrt_alphas[t].unsqueeze(1) * v_batch + sqrt_one_minus_alphas[t].unsqueeze(1) * noise
                loss = F.mse_loss(model(noisy_v_batch, t), noise)
                epoch_val_loss += loss.item() * b_sz

        avg_val_loss = epoch_val_loss / val_matrix.size(0)

        # Log combined Epoch Losses to Tensorboard Scalar Panels
        writer.add_scalar("Loss/Epoch-Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Epoch-Val", avg_val_loss, epoch)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ──────────────── GENERATION STEP & LE NET TRACKING ────────────────
        if (epoch + 1) % 5 == 0:
            print("Running full Reverse Diffusion Generation Loop to evaluate LeNet Score...")
            with torch.no_grad():
                generated_samples = torch.randn(8, total_dim, device=device)
                
                for step in reversed(range(0, num_steps)):
                    t_vec = torch.full((generated_samples.shape[0],), step, device=device, dtype=torch.long)
                    predicted_noise = model(generated_samples, t_vec)
                    
                    alpha_t = alphas[step]
                    alpha_cumprod_t = alphas_cumprod[step]
                    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas[step]
                    
                    mean = (1.0 / torch.sqrt(alpha_t)) * (generated_samples - ((betas[step] / sqrt_one_minus_alpha_cumprod_t) * predicted_noise))
                    
                    if step > 0:
                        noise_z = torch.randn_like(generated_samples)
                        generated_samples = mean + torch.sqrt(posterior_variance[step]) * noise_z
                    else:
                        generated_samples = mean

                pixel_batch = []
                coords = torch.stack(torch.meshgrid(
                    torch.linspace(-1, 1, 28), torch.linspace(-1, 1, 28), indexing="ij"
                ), dim=-1).view(-1, 2).to(device)
                
                for idx, flat_inr in enumerate(generated_samples):
                    active_inr = deserialize_weights_to_inr(flat_inr, mnist_layout).to(device).eval()
                    pixels = active_inr(coords).view(1, 28, 28)
                    pixel_batch.append(pixels)
                    
                    torch.save(flat_inr.cpu(), f"checkpoints/generated_inrs/latest_generated_inr_{idx}.pt")
                
                img_tensors = torch.stack(pixel_batch)
                
                # Compute custom LeNet Performance Score
                lenet_score = calculate_lenet_score(img_tensors, lenet_evaluator)
                print(f"✦ Generated LeNet Score Metric at Epoch {epoch+1}: {lenet_score:.4f}")
                
                # Log score to scalar graph panel
                writer.add_scalar("Metrics/LeNet-Score", lenet_score, epoch)
                
                # Log actual image visual representations into Tensorboard Image Board Grid
                img_grid = torchvision.utils.make_grid(img_tensors, nrow=4, normalize=True)
                writer.add_image("Visuals/Generated-INR-Renders", img_grid, epoch)

        # Save continuous training model updates
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'inr_shapes': mnist_layout
        }, "checkpoints/latest_backbone.pt")

    # Safely close TensorBoard connection handles upon completion
    writer.close()
    print("🏁 Training complete. Logs successfully flushed out to summary writer panels.")