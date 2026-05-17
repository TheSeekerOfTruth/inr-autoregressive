import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.dataset import INRDataset, inr_collate_fn
from src.models.gpt import INRGPT

# A clean object wrapper for your YAML configuration dictionary
class YamlConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def main():
    # 1. Load configuration from YAML file
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find configuration file at {config_path}")
        
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    config = YamlConfig(config_data)
    
    # 2. Set up device and hardware optimization flags
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using tracking device: {device}")
    
    # Enable TensorFloat-32 calculations on Ampere/Ada GPUs for faster matrix operations
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # 3. Initialize Dataset and Dataloader
    print("Loading data partitions...")
    train_dataset = INRDataset(folder_path="data/processed_inrs", split='train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=inr_collate_fn,
        pin_memory=(device == 'cuda') # Speeds up CPU-to-GPU data transfer
    )

    # 4. Initialize Model and Optimization Ecosystem
    print("Initializing INRGPT engine...")
    model = INRGPT(config).to(device)
    
    # Count parameters just to keep a pulse on the model scale
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters tracked: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        beta1=0.9, 
        beta2=0.95, 
        weight_decay=0.1
    )

    # Create a clean folder directory for saving weights checkpoints
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 5. Core Execution Loop
    print("Beginning model training execution...")
    model.train()
    
    for epoch in range(config.max_epochs):
        running_loss = 0.0
        steps_in_epoch = 0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            # Extract sequence inputs and move metadata metrics to the target hardware device
            batch_x_neurons = batch['x_neurons']
            layer_ids = batch['layer_ids'].to(device)
            
            # Step A: Run the unpadded forward representation pass
            # Returns hidden states of shape: (Batch, Sequence_Length, n_embd)
            hidden_states = model(batch_x_neurons, layer_ids)
            
            # Extract layout metadata rules and targets required for the custom decoder evaluation
            target_layer_ids = batch['target_layer_ids']
            batch_y_neurons = batch['y_neurons']
            
            B, T, _ = hidden_states.size()
            loss = 0.0
            total_elements = 0
            
            # Step B: Iterate dynamically over unpadded batches to process non-uniform vectors
            for b in range(B):
                for t in range(T):
                    pred_vector = hidden_states[b, t]
                    target_vector = batch_y_neurons[b][t].to(device)
                    target_l_id = target_layer_ids[b, t].item()

                    # Direct the uniform 384-dim prediction to the correct dimensional down-sampler
                    if target_l_id == 0:
                        logits = model.layer_0_decoder(pred_vector)   # Evaluates shapes of 3
                    else:
                        logits = model.layer_1_2_decoder(pred_vector) # Evaluates shapes of 33
                    
                    # Accumulate sums of absolute squared errors across varying dimensions
                    loss += F.mse_loss(logits, target_vector, reduction='sum')
                    total_elements += target_vector.numel()

            # Normalize the batch loss across the exact element count to protect scales
            if total_elements > 0:
                loss = loss / total_elements
            
            # Step C: Backpropagation & Optimization steps
            loss.backward()
            
            # Clip gradients at 1.0 to prevent the mathematical explosion of variance steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics Tracking
            running_loss += loss.item()
            steps_in_epoch += 1
            
            if step % config.log_interval == 0:
                print(f"Epoch {epoch+1}/{config.max_epochs} | Step {step} | Batch MSE Loss: {loss.item():.6f}")

        # Step D: Save a model state check-point at the conclusion of every completed epoch
        epoch_avg_loss = running_loss / steps_in_epoch
        print(f"--- Epoch {epoch+1} Complete | Average MSE Loss: {epoch_avg_loss:.6f} ---")
        
        checkpoint_path = os.path.join(config.checkpoint_dir, f"inr_gpt_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_avg_loss,
            'config': config_data
        }, checkpoint_path)
        print(f"Saved architectural snapshot to {checkpoint_path}\n")

    print("Training phase successfully completed!")

if __name__ == "__main__":
    main()