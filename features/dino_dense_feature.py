import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import timm
import os


class DenseDINO:

    def __init__(self, device="cuda"):

        self.device = device

        model_path = os.path.join(
            os.path.dirname(__file__),
            "/home/b224/wwmt/VFMStitch_ABUS_research/models/dinov2_vitb14_pretrain.pth"
        )

        self.model = timm.create_model(
            #"vit_large_patch14_dinov2",
            "vit_base_patch14_dinov2",
            pretrained=False
        )
        #ckpt = torch.load(model_path, map_location="cpu")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)

        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]

        self.model.load_state_dict(ckpt, strict=False)

        self.model = self.model.to(device)
        self.model.eval()

        #self.transform = T.Compose([
         #   T.Resize((224, 224)),
          #  T.ToTensor()
        #])
        self.transform = T.Compose([
        T.Resize(518),
        T.CenterCrop(518),
        T.ToTensor()
    ])

    def extract_slice(self, img):

        img = Image.fromarray(img).convert("RGB")

        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model.forward_features(img)
            #print(type(feat), feat.shape)
            #feat = feat["x_norm_patchtokens"]
            feat = feat[:,1:,:]
            feat = feat.reshape(1,37,37,-1)

        return feat.cpu().numpy()[0]

    def extract_volume(self, volume):

        feats = []


        for i in range(volume.shape[0]):

            slice_img = volume[i, :, :]

            feats.append(self.extract_slice(slice_img))

        return np.array(feats)