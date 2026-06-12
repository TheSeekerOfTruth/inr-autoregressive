import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call, vmap
from model import mnistINR


def get_device(device=None):
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(device=None):
    return mnistINR().to(get_device(device))


def base_checkpoint_path(base_dir, digit):
    return os.path.join(base_dir, f"digit_{digit}_base.pth")

def load_inr(path, device=None):
    """Reconstruct a trained INR from a saved ``state_dict``."""
    device = get_device(device)
    model = make_model(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def get_or_train_base(dataset, digit, position, base_dir, device=None,
                      coords=None, **train_kwargs):
    """Sequential fallback/cache loader for the baseline starting weights."""
    device = get_device(device)
    os.makedirs(base_dir, exist_ok=True)
    path = base_checkpoint_path(base_dir, digit)

    model = make_model(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        return model, {"cached": True, "epochs": None, "final_loss": None}

    if coords is None:
        coords = dataset.coordinate_grid().to(device)
    target, _ = dataset.get_target(position)
    target = target.to(device)

    # Simple single INR fit for base
    optimizer = optim.Adam(model.parameters(), lr=train_kwargs.get('lr', 1e-3))
    criterion = nn.MSELoss()
    max_epochs = train_kwargs.get('max_epochs', 500)
    loss_threshold = train_kwargs.get('loss_threshold', 0.002)

    epoch = 1
    while True:
        optimizer.zero_grad()
        loss = criterion(model(coords), target)
        loss_val = loss.item()
        if loss_val < loss_threshold or epoch > max_epochs:
            break
        loss.backward()
        optimizer.step()
        epoch += 1

    torch.save(model.state_dict(), path)
    return model, {"cached": False, "epochs": epoch, "final_loss": loss_val}


def train_digit_inrs_batched(dataset, digit, positions, save_dir, base_dir, device=None,
                             coords=None, batch_size=128, lr=1e-3, max_epochs=500,
                             loss_threshold=0.002, **train_kwargs):
    """Vectorized trainer executing warm-start configurations in parallel batches.
    
    Batches independent parameter matrices together to maximize GPU utilization.
    """
    device = get_device(device)
    if coords is None:
        coords = dataset.coordinate_grid().to(device)

    digit_dir = os.path.join(save_dir, f"digit_{digit}")
    os.makedirs(digit_dir, exist_ok=True)

    # 1. Establish structural Base State Checkpoint
    base_model, base_stats = get_or_train_base(
        dataset, digit, positions[0], base_dir, device=device, coords=coords,
        lr=lr, max_epochs=max_epochs, loss_threshold=loss_threshold, **train_kwargs
    )
    base_state = base_model.state_dict()

    base_ckpt = os.path.join(digit_dir, "sample_0.pth")
    torch.save(base_state, base_ckpt)
    
    records = [{
        "digit": digit, "index": 0, "position": int(positions[0]),
        "epochs": base_stats.get("epochs"), "final_loss": base_stats.get("final_loss"),
        "checkpoint": base_ckpt, "is_base": True, "base_cached": base_stats.get("cached"),
    }]

    # Define a clean, stateless target template
    blueprint_model = make_model(device)
    buffers = {k: v.to(device) for k, v in blueprint_model.named_buffers()}

    remaining_positions = positions[1:]
    total_remaining = len(remaining_positions)

    # 2. Iterate through data arrays using explicit parallel chunks
    for batch_idx in range(0, total_remaining, batch_size):
        chunk_positions = remaining_positions[batch_idx : batch_idx + batch_size]
        curr_batch_len = len(chunk_positions)

        # Build dynamic target stacks
        targets = torch.stack([dataset.get_target(pos)[0].to(device) for pos in chunk_positions])

        # Convert shared base model state dict arrays into batched parallel fields
        batched_params = {}
        for k, v in base_state.items():
            batched_params[k] = (
                v.clone()
                .detach()
                .to(device)
                .repeat(curr_batch_len, *([1] * v.dim()))
                .requires_grad_(True)
            )

        # Functional single loss configuration mapped out for vmap execution
        def compute_loss_single(params, buffers, coords, single_target):
            predictions = functional_call(blueprint_model, (params, buffers), coords)
            return torch.nn.functional.mse_loss(predictions, single_target)

        # Construct vectorized loss evaluation function mapping out dimension 0 bounds
        batched_loss_fn = vmap(compute_loss_single, in_dims=(0, None, None, 0))
        optimizer = optim.Adam(batched_params.values(), lr=lr)

        # 3. Vectorized Training Execution Phase
        epoch = 1
        while True:
            optimizer.zero_grad()
            losses = batched_loss_fn(batched_params, buffers, coords, targets)
            mean_loss = losses.mean()
            
            # Optimization condition check bounds
            if mean_loss.item() < loss_threshold or epoch > max_epochs:
                break
                
            mean_loss.backward()
            optimizer.step()
            epoch += 1

        # 4. Extract parameters and distribute individual state check-pointing targets
        for b in range(curr_batch_len):
            global_idx = batch_idx + b + 1
            
            single_state = {k: v[b].detach().cpu() for k, v in batched_params.items()}
            ckpt = os.path.join(digit_dir, f"sample_{global_idx}.pth")
            torch.save(single_state, ckpt)

            records.append({
                "digit": digit, "index": global_idx, "position": int(chunk_positions[b]),
                "epochs": epoch, "final_loss": float(losses[b].item()),
                "checkpoint": ckpt, "is_base": False, "base_cached": None,
            })

        print(f"  digit {digit}: {min(batch_idx + batch_size, total_remaining)}/{total_remaining} INRs trained (Batched)")

    return records


def train_all_digits(dataset, save_dir="checkpoints/all_inrs",
                     base_dir="checkpoints/base_inrs", device=None,
                     digits=range(10), samples_per_digit=None, coords=None,
                     batch_size=1024, **train_kwargs):
    """Vectorized end-to-end framework processing requested MNIST dataset labels."""
    device = get_device(device)
    os.makedirs(save_dir, exist_ok=True)
    if coords is None:
        coords = dataset.coordinate_grid().to(device)

    groups = dataset.positions_by_label()
    all_records = []
    
    for digit in digits:
        positions = groups[digit]
        if samples_per_digit is not None:
            positions = positions[:samples_per_digit]

        print(f"=== Vectorized Digit {digit}: {len(positions)} images ===")
        records = train_digit_inrs_batched(
            dataset, digit, positions, save_dir, base_dir, device=device,
            coords=coords, batch_size=batch_size, **train_kwargs
        )
        all_records.extend(records)

    return all_records


def reconstruct(model, coords, side=28, device=None):
    device = get_device(device)
    with torch.no_grad():
        prediction = model(coords.to(device)).cpu().reshape(side, side)
    return prediction