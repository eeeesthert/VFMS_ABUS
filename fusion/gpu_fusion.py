import torch
import numpy as np

def gpu_fuse(volumes, transforms):

    device = "cuda"

    fused = None

    for v in volumes:

        data = torch.tensor(v.data, device=device, dtype=torch.float32)

        if fused is None:
            fused = data
        else:
            fused = torch.maximum(fused, data)

    return fused.cpu().numpy()