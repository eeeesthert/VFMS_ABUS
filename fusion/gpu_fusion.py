import torch
import torch.nn.functional as F
import numpy as np


# ----------------------------------------------------
# weight map (for weighted blending)
# ----------------------------------------------------

def compute_weight_map(shape):

    D,H,W = shape

    z = np.minimum(np.arange(D), np.arange(D)[::-1])
    y = np.minimum(np.arange(H), np.arange(H)[::-1])
    x = np.minimum(np.arange(W), np.arange(W)[::-1])

    zz,yy,xx = np.meshgrid(z,y,x,indexing="ij")

    weight = np.minimum(np.minimum(zz,yy),xx).astype(np.float32)

    weight += 1e-3
    weight /= weight.max()

    return weight


# ----------------------------------------------------
# volume corner
# ----------------------------------------------------

def get_volume_corners(shape):

    D,H,W = shape

    corners = np.array([
        [0,0,0],
        [0,0,W],
        [0,H,0],
        [0,H,W],
        [D,0,0],
        [D,0,W],
        [D,H,0],
        [D,H,W],
    ],dtype=np.float32)

    return corners


# ----------------------------------------------------
# compute global bounding box
# ----------------------------------------------------

def compute_global_bbox(volumes, transforms):

    all_pts = []

    for vol,T in zip(volumes,transforms):

        corners = get_volume_corners(vol.data.shape)

        corners = np.concatenate([corners,np.ones((8,1))],axis=1)

        warped = (T @ corners.T).T[:,:3]

        all_pts.append(warped)

    all_pts = np.vstack(all_pts)

    min_pt = np.floor(all_pts.min(axis=0)).astype(int)
    max_pt = np.ceil(all_pts.max(axis=0)).astype(int)

    return min_pt,max_pt


# ----------------------------------------------------
# grid generator
# ----------------------------------------------------

def build_grid(shape,device):

    D,H,W = shape

    z = torch.linspace(-1,1,D,device=device)
    y = torch.linspace(-1,1,H,device=device)
    x = torch.linspace(-1,1,W,device=device)

    zz,yy,xx = torch.meshgrid(z,y,x,indexing="ij")

    grid = torch.stack((xx,yy,zz),dim=-1)

    grid = grid.unsqueeze(0)

    return grid


# ----------------------------------------------------
# warp volume
# ----------------------------------------------------

def warp_volume(volume,T,out_shape,offset,device):

    D,H,W = out_shape

    vol = torch.tensor(volume,dtype=torch.float32,device=device)
    vol = vol.unsqueeze(0).unsqueeze(0)

    grid = build_grid(out_shape,device)

    coords = grid.view(-1,3)

    R = torch.tensor(T[:3,:3],dtype=torch.float32,device=device)
    t = torch.tensor(T[:3,3]-offset,dtype=torch.float32,device=device)

    coords = torch.matmul(coords,R.T)+t

    coords = coords.view(1,D,H,W,3)

    warped = F.grid_sample(
        vol,
        coords,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )

    return warped[0,0]


# ----------------------------------------------------
# main fusion
# ----------------------------------------------------

def gpu_fuse(volumes,transforms):

    device="cuda"

    # ------------------------------------------------
    # compute global bbox
    # ------------------------------------------------

    min_pt,max_pt = compute_global_bbox(volumes,transforms)

    size = max_pt-min_pt

    print("Global FOV:",size)

    fused = torch.zeros(size.tolist(),device=device)
    weight_sum = torch.zeros(size.tolist(),device=device)

    # ------------------------------------------------
    # fuse each volume
    # ------------------------------------------------

    for vol,T in zip(volumes,transforms):

        data = vol.data

        warped = warp_volume(
            data,
            T,
            size,
            min_pt,
            device
        )

        weight = torch.tensor(
            compute_weight_map(data.shape),
            device=device
        )

        weight = warp_volume(
            weight.cpu().numpy(),
            T,
            size,
            min_pt,
            device
        )

        fused += warped*weight
        weight_sum += weight

    fused = fused/(weight_sum+1e-6)

    return fused.cpu().numpy()