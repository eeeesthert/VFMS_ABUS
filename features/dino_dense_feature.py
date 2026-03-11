import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import timm
import os
from sklearn.decomposition import PCA


class DenseDINO:

    def __init__(self, device="cuda:1"):

        self.device = self._resolve_device(device)

        model_path = "/home/b224/wwmt/VFMStitch_ABUS_research/models/dinov2_vitb14_pretrain.pth"

        self.model = timm.create_model(
            "vit_base_patch14_dinov2",
            pretrained=False
        )

        ckpt = torch.load(model_path, map_location="cpu")

        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]

        self.model.load_state_dict(ckpt, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize(518),
            T.CenterCrop(518),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)
            )
        ])

    def _resolve_device(self, device):

        if not device.startswith("cuda"):
            return device

        if not torch.cuda.is_available():
            print("CUDA unavailable, fallback to CPU")
            return "cpu"

        if device == "cuda":
            return "cuda:0"

        if ":" in device:
            idx = int(device.split(":", 1)[1])
            if idx < torch.cuda.device_count():
                return device
            print(f"{device} unavailable, fallback to cuda:0")
            return "cuda:0"

        return device
    def normalize_slice(self, img):

        img = img.astype(np.float32)

        img = img - img.min()

        if img.max() > 0:
            img = img / img.max()

        img = (img * 255).astype(np.uint8)

        return img


    def pca_reduce(self, feat_map, dim=16):

        h,w,c = feat_map.shape

        flat = feat_map.reshape(-1,c)

        pca = PCA(n_components=dim)

        flat = pca.fit_transform(flat)

        return flat.reshape(h,w,dim)


    def extract_volume(self, volume, batch_size=32):

        slices = []

        for i in range(volume.shape[0]):

            img = volume[i,:,:]

            img = self.normalize_slice(img)

            img = Image.fromarray(img).convert("RGB")

            img = self.transform(img)

            slices.append(img)

        slices = torch.stack(slices).to(self.device)

        feats = []

        with torch.no_grad():

            for i in range(0, slices.shape[0], batch_size):

                batch = slices[i:i+batch_size]

                feat = self.model.forward_features(batch)

                tokens = feat[:,1:,:]

                B,N,C = tokens.shape

                grid = int(np.sqrt(N))

                tokens = tokens.reshape(B,grid,grid,C)

                feats.append(tokens.cpu().numpy())

        feats = np.concatenate(feats,axis=0)

        feats_reduced = []

        for f in feats:

            feats_reduced.append(self.pca_reduce(f))

        return np.array(feats_reduced)
