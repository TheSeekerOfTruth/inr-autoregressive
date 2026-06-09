"""Reusable logic for fitting INRs to MNIST images with per-digit warm-starting.

Strategy
--------
For each digit we train one "base" INR from scratch on that digit's first image.
That base INR is then used to *initialize* (warm-start) the INRs for every other
image of the same digit, which converge much faster than training from scratch.

The base INR is cached on disk (``base_dir``) so it is trained only once and
reused across runs, keeping the warm-start initialization stable/reproducible.

All functions are import-friendly: run notebooks from ``data_generation/`` so the
``model`` and ``dataset`` modules resolve, matching the existing ``train.py``.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from model import mnistINR


def get_device(device=None):
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(device=None):
    return mnistINR().to(get_device(device))


def train_inr(model, coords, target, lr=1e-3, loss_threshold=0.002,
              max_epochs=10000, log_every=None, return_history=False):
    """Fit a single INR to one image.

    Mirrors the loop in ``train.py``: optimize until the MSE drops below
    ``loss_threshold`` or ``max_epochs`` is exceeded.

    Returns a dict with ``epochs`` and ``final_loss`` (and ``history`` if
    ``return_history`` is set).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = [] if return_history else None
    epoch = 1
    while True:
        optimizer.zero_grad()
        prediction = model(coords)
        loss = criterion(prediction, target)
        loss_value = loss.item()
        if return_history:
            history.append(loss_value)

        if loss_value < loss_threshold or epoch > max_epochs:
            break

        loss.backward()
        optimizer.step()

        if log_every and epoch % log_every == 0:
            print(f"    Epoch {epoch} | Loss: {loss_value:.6f}")
        epoch += 1

    result = {"epochs": epoch, "final_loss": loss_value}
    if return_history:
        result["history"] = history
    return result


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
    """Return the base INR for ``digit``: load it from ``base_dir`` if cached,
    otherwise train it on the image at ``position`` and save it.

    Returns ``(model, stats)`` where ``stats`` includes ``cached`` (bool).
    """
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

    stats = train_inr(model, coords, target, **train_kwargs)
    torch.save(model.state_dict(), path)
    stats["cached"] = False
    return model, stats


def train_digit_inrs(dataset, digit, positions, save_dir, base_dir, device=None,
                     coords=None, log_every_samples=200, **train_kwargs):
    """Train INRs for every image of ``digit`` listed in ``positions``.

    ``positions[0]`` becomes the base INR (trained once / loaded from cache);
    every remaining image is warm-started from that base. Each INR's
    ``state_dict`` is saved to ``save_dir/digit_<d>/sample_<i>.pth``.

    Returns one record dict per trained INR.
    """
    device = get_device(device)
    if coords is None:
        coords = dataset.coordinate_grid().to(device)

    digit_dir = os.path.join(save_dir, f"digit_{digit}")
    os.makedirs(digit_dir, exist_ok=True)

    # 1. Base INR = first image of this digit (train once, cache & reuse).
    base_model, base_stats = get_or_train_base(
        dataset, digit, positions[0], base_dir, device=device, coords=coords,
        **train_kwargs)
    base_state = base_model.state_dict()

    base_ckpt = os.path.join(digit_dir, "sample_0.pth")
    torch.save(base_state, base_ckpt)
    records = [{
        "digit": digit, "index": 0, "position": int(positions[0]),
        "epochs": base_stats.get("epochs"), "final_loss": base_stats.get("final_loss"),
        "checkpoint": base_ckpt, "is_base": True, "base_cached": base_stats.get("cached"),
    }]

    # 2. Warm-start every remaining image from the base INR.
    total = len(positions)
    for i, position in enumerate(positions[1:], start=1):
        model = make_model(device)
        model.load_state_dict(base_state)

        target, _ = dataset.get_target(position)
        target = target.to(device)
        stats = train_inr(model, coords, target, **train_kwargs)

        ckpt = os.path.join(digit_dir, f"sample_{i}.pth")
        torch.save(model.state_dict(), ckpt)
        records.append({
            "digit": digit, "index": i, "position": int(position),
            "epochs": stats["epochs"], "final_loss": stats["final_loss"],
            "checkpoint": ckpt, "is_base": False, "base_cached": None,
        })

        if log_every_samples and (i + 1) % log_every_samples == 0:
            print(f"  digit {digit}: {i + 1}/{total} INRs trained")

    return records


def train_all_digits(dataset, save_dir="checkpoints/all_inrs",
                     base_dir="checkpoints/base_inrs", device=None,
                     digits=range(10), samples_per_digit=None, coords=None,
                     log_every_samples=200, **train_kwargs):
    """Full pipeline: fit warm-started INRs for every requested digit.

    Parameters
    ----------
    samples_per_digit : int or None
        Cap on images per digit (handy for smoke tests). ``None`` trains every
        available image of each digit.

    Returns one flat list of per-INR record dicts across all digits.
    """
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

        print(f"=== Digit {digit}: {len(positions)} images ===")
        records = train_digit_inrs(
            dataset, digit, positions, save_dir, base_dir, device=device,
            coords=coords, log_every_samples=log_every_samples, **train_kwargs)
        all_records.extend(records)

    return all_records


def reconstruct(model, coords, side=28, device=None):
    """Render an INR over ``coords`` into a ``side x side`` image tensor (CPU)."""
    device = get_device(device)
    with torch.no_grad():
        prediction = model(coords.to(device)).cpu().reshape(side, side)
    return prediction
