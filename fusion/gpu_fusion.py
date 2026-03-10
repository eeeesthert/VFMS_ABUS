import torch
import torch.nn.functional as F
import numpy as np


def compute_weight_map(shape):

    D, H, W = shape

    z = np.minimum(np.arange(D), np.arange(D)[::-1])
    y = np.minimum(np.arange(H), np.arange(H)[::-1])
    x = np.minimum(np.arange(W), np.arange(W)[::-1])

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    weight = np.minimum(np.minimum(zz, yy), xx).astype(np.float32)

    weight += 1e-3
    weight /= weight.max()

    return weight


def build_grid(shape, device):

    D, H, W = shape

    z = torch.linspace(-1, 1, D, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)

    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    grid = torch.stack((xx, yy, zz), dim=-1)

    grid = grid.unsqueeze(0)

    return grid


def apply_transform(volume, T, device):

    D, H, W = volume.shape

    volume = torch.tensor(volume, dtype=torch.float32, device=device)

    volume = volume.unsqueeze(0).unsqueeze(0)

    grid = build_grid((D, H, W), device)

    R = torch.tensor(T[:3, :3], dtype=torch.float32, device=device)
    t = torch.tensor(T[:3, 3], dtype=torch.float32, device=device)

    coords = grid.view(-1, 3)

    coords = torch.matmul(coords, R.T) + t

    coords = coords.view(1, D, H, W, 3)

    warped = F.grid_sample(
        volume,
        coords,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )

    warped = warped[0, 0]

    return warped


def gpu_fuse(volumes, transforms):

    device = "cuda"

    fused = None
    weight_sum = None

    for v, T in zip(volumes, transforms):

        warped = apply_transform(v.data, T, device)

        weight = compute_weight_map(v.data.shape)

        weight = torch.tensor(weight, device=device)

        if fused is None:

            fused = warped * weight
            weight_sum = weight

        else:

            fused += warped * weight
            weight_sum += weight

    fused = fused / (weight_sum + 1e-6)

    return fused.cpu().numpy()