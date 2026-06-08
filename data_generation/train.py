import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataset import OptimizedDataset
from model import mnistINR


save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

mnist_dataset = OptimizedDataset(as_inr=True)
prev_weights = None
num_experiments = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for i in range(num_experiments):    
    # Instantiate the model
    model = mnistINR().to(device)
    
    # If we have weights from a previous run, use them to initialize
    if prev_weights is not None:
        print("Warm-starting with previous model weights...")
        model.load_state_dict(prev_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    coords, target, label = mnist_dataset[i]
    coords = coords.to(device)
    target = target.to(device)
    print(f"EXPERIMENT: Sample #{i} | Digit: {label}")

    # Simple training loop
    finish = False
    epoch = 1
    while(not finish):
        optimizer.zero_grad()
        prediction = model(coords)
        loss = criterion(prediction, target)
        if(loss < 0.002 or epoch > 10000):
            finish = True
            break
        loss.backward()
        optimizer.step()
        epoch = epoch + 1
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
            
    # Save weights for the next iteration
    save_path = os.path.join(save_dir, f"model_sample_{i}_digit_{label}.pth")
    torch.save(model.state_dict(), save_path)
    prev_weights = model.state_dict()
    print(f"-> Finished Sample #{i}. Saved to '{save_path}'")