import torch


def make_coordinates(img_shape=(28, 28)):
    h, w = img_shape
    ys = torch.linspace(-1, 1, h)
    xs = torch.linspace(-1, 1, w)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


def render_inr(state_dict, img_shape=(28, 28), w0=30.0):
    """Forward-pass a SIREN state_dict to produce an image tensor (H, W).

    Matches the INR architecture from nn/inr.py: Sine(w0) activations on all
    hidden layers, nn.Linear output, and a +0.5 offset (see INR.forward).
    """
    coords = make_coordinates(img_shape)
    x = coords
    layer_indices = sorted(set(int(k.split(".")[1]) for k in state_dict if "seq" in k))
    for i, idx in enumerate(layer_indices):
        W = state_dict[f"seq.{idx}.weight"]
        b = state_dict[f"seq.{idx}.bias"]
        x = x @ W.T + b
        if i < len(layer_indices) - 1:
            x = torch.sin(w0 * x)
    return (x + 0.5).reshape(img_shape)
