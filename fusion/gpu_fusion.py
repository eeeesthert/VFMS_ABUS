import torch
import torch.nn.functional as F
import numpy as np


def compute_weight_map(shape):
    d, h, w = shape
    z = np.minimum(np.arange(d), np.arange(d)[::-1])
    y = np.minimum(np.arange(h), np.arange(h)[::-1])
    x = np.minimum(np.arange(w), np.arange(w)[::-1])
    zz,yy,xx = np.meshgrid(z,y,x,indexing="ij")

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    weight = np.minimum(np.minimum(zz, yy), xx).astype(np.float32)

    weight += 1e-3
    # weight /= weight.max()
    weight /= (weight.max() + 1e-8)

    return weight



def get_volume_corners(shape):

    d, h, w = shape
    return np.array([
        [0, 0, 0],
        [0, 0, w - 1],
        [0, h - 1, 0],
        [0, h - 1, w - 1],
        [d - 1, 0, 0],
        [d - 1, 0, w - 1],
        [d - 1, h - 1, 0],
        [d - 1, h - 1, w - 1],
    ], dtype=np.float32)



def compute_global_bbox(volumes, transforms):
    all_pts = []

    for vol, t in zip(volumes, transforms):

        corners = get_volume_corners(vol.data.shape)

        # corners = np.concatenate([corners,np.ones((8,1))],axis=1)

        # warped = (T @ corners.T).T[:,:3]
        corners_h = np.concatenate([corners, np.ones((8, 1), dtype=np.float32)], axis=1)
        warped = (t @ corners_h.T).T[:, :3]

        all_pts.append(warped)

    all_pts = np.vstack(all_pts)

    min_pt = np.floor(all_pts.min(axis=0)).astype(np.int32)
    max_pt = np.ceil(all_pts.max(axis=0)).astype(np.int32)
    return min_pt, max_pt


def _build_world_grid(out_shape, offset, device):
    d, h, w = out_shape
    z = torch.arange(d, dtype=torch.float32, device=device) + float(offset[0])
    y = torch.arange(h, dtype=torch.float32, device=device) + float(offset[1])
    x = torch.arange(w, dtype=torch.float32, device=device) + float(offset[2])
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([zz, yy, xx], dim=-1)

def _normalize_to_grid(coords_zyx, src_shape):
    d, h, w = src_shape
    z = coords_zyx[..., 0]
    y = coords_zyx[..., 1]
    x = coords_zyx[..., 2]

    if d > 1:
        z = 2.0 * z / (d - 1) - 1.0
    else:
        z = torch.zeros_like(z)
    if h > 1:
        y = 2.0 * y / (h - 1) - 1.0
    else:
        y = torch.zeros_like(y)
    if w > 1:
        x = 2.0 * x / (w - 1) - 1.0
    else:
        x = torch.zeros_like(x)
    
    # zz,yy,xx = torch.meshgrid(z,y,x,indexing="ij")
    return torch.stack([x, y, z], dim=-1)

def warp_volume(volume, transform, out_shape, global_min_pt, device):
    src = torch.tensor(volume, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    world_grid = _build_world_grid(out_shape, global_min_pt, device)
    ones = torch.ones((*out_shape, 1), dtype=torch.float32, device=device)
    world_h = torch.cat([world_grid, ones], dim=-1).reshape(-1, 4)

    t_inv = torch.tensor(np.linalg.inv(transform), dtype=torch.float32, device=device)
    src_coords = (world_h @ t_inv.T)[:, :3].reshape(*out_shape, 3)

    sample_grid = _normalize_to_grid(src_coords, volume.shape).unsqueeze(0)

    

    warped = F.grid_sample(
         src,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped[0, 0]

def gpu_fuse(volumes, transforms):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    min_pt, max_pt = compute_global_bbox(volumes, transforms)
    size = (max_pt - min_pt + 1).astype(np.int32)
    print("Global FOV:", size)
    fused = torch.zeros(size.tolist(), dtype=torch.float32, device=device)
    weight_sum = torch.zeros(size.tolist(), dtype=torch.float32, device=device)
    for vol, t in zip(volumes, transforms):
        data = vol.data
        warped = warp_volume(data, t, size, min_pt, device)
        weight = compute_weight_map(data.shape)
        warped_weight = warp_volume(weight, t, size, min_pt, device)
        fused += warped * warped_weight
        weight_sum += warped_weight
    fused = fused / (weight_sum + 1e-6)
    return fused.cpu().numpy()

