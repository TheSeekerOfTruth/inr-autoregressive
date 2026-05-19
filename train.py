import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from src.data.dataset import INRDataset
from src.models.gpt import INRGPT

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
            x_neurons = batch['x_neurons'].to(device)
            layer_ids = batch['layer_ids'].to(device)
            target_layer_ids = batch['target_layer_ids'].to(device)
            y_neurons = batch['y_neurons'].to(device)
            
            predictions = model(x_neurons, layer_ids, target_layer_ids)
            loss = F.mse_loss(predictions, y_neurons)
            
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
    # This creates a 'runs/' directory automatically
    writer = SummaryWriter(log_dir="runs/inr_gpt_experiment")
    
    # 2. Hardware acceleration setups
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # 3. Load Dataset & Create Train/Validation Splits
    print("Loading data and configuring splits...")
    full_dataset = INRDataset(folder_path="data/processed_inrs_1", split='train')
    
    # Automatically allocate 90% for training and 10% for tracking validation loss
    # train_size = int(0.9 * len(full_dataset))
    train_size = 1 
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=(device == 'cuda')
    )
    #val_loader = DataLoader(
    #    val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=(device == 'cuda')
    #)

    # 4. Initialize Model and Optimizer
    print("Initializing INRGPT engine...")
    model = INRGPT(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters tracked: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, 
        betas=(0.9, 0.95), weight_decay=0.1
    )

    # Dictionary history tracker for our visual Jupyter Notebook dashboard
    history_logs = []

    # 5. Core Execution Loop
    print("Beginning model training execution...")
    global_step = 0

    for epoch in range(config.max_epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        train_steps = 0
        
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            # Move everything cleanly to the active hardware device
            x_neurons = batch['x_neurons'].to(device)
            layer_ids = batch['layer_ids'].to(device)
            target_layer_ids = batch['target_layer_ids'].to(device)
            y_neurons = batch['y_neurons'].to(device)
            
            predictions = model(x_neurons, layer_ids, target_layer_ids)
            
            loss = F.mse_loss(predictions, y_neurons)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item()
            train_steps += 1
            
            if global_step % config.log_interval == 0:
                #current_val_loss = evaluate_sub_loss(model, val_loader, device)
                writer.add_scalars("Loss/Batch_Step", {
                    "Train": loss.item(),
                    #"Validation": current_val_loss
                }, global_step)
            global_step += 1

        epoch_train_loss = running_train_loss / train_steps

        # --- VALIDATION PHASE ---
        #model.eval()
        #running_val_loss = 0.0
        #val_steps = 0
        
        #with torch.no_grad():
        #    for batch in val_loader:
        #        x_neurons = batch['x_neurons'].to(device)
        #        layer_ids = batch['layer_ids'].to(device)
        #        target_layer_ids = batch['target_layer_ids'].to(device)
        #        y_neurons = batch['y_neurons'].to(device)
                
        #        predictions = model(x_neurons, layer_ids, target_layer_ids)
        #        loss = F.mse_loss(predictions, y_neurons)
                
        #        running_val_loss += loss.item()
        #        val_steps += 1
                
        #epoch_val_loss = running_val_loss / val_steps
        #print(f"--- Epoch {epoch+1} Complete | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} ---")
        
    # 6. Save Final Checkpoint
    checkpoint_path = os.path.join(config.checkpoint_dir, f"inr_gpt.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config' : config
    }, checkpoint_path)

    print("Training phase successfully completed!")

if __name__ == "__main__":
    main()