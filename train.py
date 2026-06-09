import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from src.data.dataset import INRDataset
from src.models.gpt import INRGPT
from torch.amp import autocast 
import math

def get_lr_lambda(current_step, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

class YamlConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def evaluate_sub_loss(model, val_loader, device):
    model.eval()
    total_val_loss = 0.0
    actual_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # 🔥 OPTIMIZATION: Asynchronous GPU data transfer
            x_tokens = batch['x_tokens'].to(device, non_blocking=True)
            y_tokens = batch['y_tokens'].to(device, non_blocking=True)
            
            # 🔥 OPTIMIZATION: Run validation inference in bfloat16
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x_tokens, y_tokens)            
            total_val_loss += loss.item()
            actual_batches += 1
    model.train() 
    return total_val_loss / max(actual_batches, 1)

def main():
    # 1. Load configuration from YAML file
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find configuration file at {config_path}")
        
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    config = YamlConfig(config_data)
    
    # Set up directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    # 2. Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir="runs/inr_gpt_experiment")
    
    # 2. Hardware acceleration setups
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        # You had this! This is great—it ensures Blackwell uses TF32 for any leftover float32 math
        torch.set_float32_matmul_precision('high')

    # 3. Load Dataset & Create Train/Validation Splits
    print("Loading data and configuring splits...")
    full_dataset = INRDataset(folder_path="data/processed_inrs", split='train')
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # ==========================================
    # 🛠️ DEBUG MODE TOGLE
    # ==========================================
    """ RUN_MINI_EXPERIMENT = True  # Set to False when you are ready for the 3-hour full run!

    if RUN_MINI_EXPERIMENT:
        print("⚠️ Running in MINI-EXPERIMENT mode with 10,000 training samples...")
        # Create a list of indices from 0 to 999
        mini_indices = list(range(10000))
        # Slice out exactly 1,000 samples for training
        train_dataset = Subset(train_dataset, mini_indices)
        
        # Slice out a smaller validation set too (e.g., 1000 samples) so validation is instant
        val_indices = list(range(1000))
        val_dataset = Subset(val_dataset, val_indices) """
    # ==========================================

    # Added num_workers to utilize CPU threads for pipeline loading alongside pin_memory
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=(device == 'cuda'), num_workers=6, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=(device == 'cuda'), num_workers=6
    )

    # 4. Initialize Model and Optimizer
    print("Initializing INRGPT engine...")
    model = INRGPT(config).to(device)

    # 🔥 OPTIMIZATION: Compile the model for Compute Capability 12.0
    # This will fuse kernels and optimize graph paths across your Blackwell layout.
    #print("Compiling model for Blackwell (max-autotune)... First batch will take a moment to warm up.")
    #model = torch.compile(raw_model, mode="max-autotune")

    # Pass the compiled model parameters to the optimizer setup
    optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=config.learning_rate, betas=(0.9,0.95), device_type=device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[900], 
        gamma=0.5
    )
    history_logs = [] 

    # 5. Core Execution Loop
    print("Beginning model training execution...")
    global_step = 0

    for epoch in range(config.max_epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        train_steps = 0
        last_lr = None
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            # 🔥 OPTIMIZATION: non_blocking=True unlocks DMA streaming across PCIe lanes
            x_tokens = batch['x_tokens'].to(device, non_blocking=True)
            y_tokens = batch['y_tokens'].to(device, non_blocking=True)
            
            # 🔥 OPTIMIZATION: AMP context block. This leverages 5th-Gen Tensor Cores.
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x_tokens, y_tokens)            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
    
            # Check if the LR has changed since the last step
            if last_lr is not None and current_lr != last_lr:
               print(f"Step {step}: Learning Rate changed from {last_lr:.2e} to {current_lr:.2e}")
            last_lr = current_lr

            running_train_loss += loss.item()
            train_steps += 1
            
            if global_step % config.log_interval == 0:
                current_val_loss = evaluate_sub_loss(model, val_loader, device)
                writer.add_scalars("Loss/Batch_Step", {
                    "Train": loss.item(),
                    "Validation": current_val_loss
                }, global_step)
            global_step += 1

        epoch_train_loss = running_train_loss / train_steps

        # --- VALIDATION PHASE (End of Epoch) ---
        model.eval()
        running_val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 🔥 OPTIMIZATION: Non-blocking moves & Autocast during main validation sweep
                x_neurons = batch['x_tokens'].to(device, non_blocking=True)
                layer_ids = batch['y_tokens'].to(device, non_blocking=True)
                
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x_neurons, layer_ids)

                running_val_loss += loss.item()
                val_steps += 1
                
        epoch_val_loss = running_val_loss / val_steps
        print(f"--- Epoch {epoch+1} Complete | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} ---")
        
    # 6. Save Final Checkpoint
    checkpoint_path = os.path.join(config.checkpoint_dir, f"inr_gpt.pt")
    
    # ⚠️ CRITICAL NOTE FOR TORCH.COMPILE:
    # When saving weights, always reference the underlying raw model ('raw_model.state_dict()').
    # Saving 'model.state_dict()' directly saves compiled hook artifacts, which will break loading later!
    torch.save({
        'model_state_dict': model.state_dict(),
        'config' : config
    }, checkpoint_path)

    print("Training phase successfully completed!")

if __name__ == "__main__":
    main()