import cv2
import numpy as np
import pathlib
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List, Optional, Tuple, Union


class Dino_extractor():
    """
    Class to extract DINO features from an image.
    """
    
    def __init__(self, method: str, cpt_path: str):
        
        assert method in ["roma", "kaiko"], "method must be one of ['roma', 'kaiko']"
        self.method = method
               
        if self.method in ["roma", "kaiko"]:
            # Load model
            sys.path.append("/detectors/RoMa")
            from romatch.models.transformer.dinov2 import vit_large
            vit_kwargs = dict(img_size= 518,
                patch_size= 14,
                init_values = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
            )
            self.dinov2_vitl14 = vit_large(**vit_kwargs)
            
            # Fetch weights
            if cpt_path is not None:
                if cpt_path.startswith("http"):
                    dinov2_weights = torch.hub.load_state_dict_from_url(cpt_path, map_location="cpu")
                elif pathlib.Path(cpt_path).exists():
                    dinov2_weights = torch.load(cpt_path, map_location="cpu")
                else:
                    raise ValueError(f"cpt_path {cpt_path} is not a valid path or url")
            else:
                dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        
        self.dinov2_vitl14.load_state_dict(dinov2_weights, strict=False)
        self.dinov2_vitl14.eval()
        self.dinov2_vitl14 = self.dinov2_vitl14.to("cuda")
        
        return
    
    def extract(self, image: np.ndarray, batched: bool = False) -> np.ndarray:
        
        if batched:
            return self._forward_batch(image)
        else:
            return self._forward(image)
    
    def _forward(self, image: np.ndarray) -> np.ndarray:
        
        image = self._preprocess_image(image)
        C, B, H, W = image.shape
        
        with torch.no_grad():
            
            if self.method in ["roma", "kaiko"]:
                features = self.dinov2_vitl14.forward_features(image)
                features = features["x_norm_patchtokens"].squeeze().reshape(H//14, W//14, -1)
            
            features = features.cpu().numpy()
            
            del image
            torch.cuda.empty_cache()
        
        return features
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        
        assert len(image.shape) == 3, "only use rgb images"
        
        if self.method == "kaiko":
            return self._preprocess_kaiko(image)
        elif self.method in ["roma"]:
            return self._preprocess_roma(image)

    def _preprocess_roma(self, image: np.ndarray) -> torch.Tensor:
        
        DINO_HW = 1036
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        # Resize and convert to tensor
        image = cv2.resize(image, (DINO_HW, DINO_HW))
        image = ((image/255) - MEAN) / STD
        image = image[None, ...].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).float().to("cuda")
        
        return image
    
    def _preprocess_kaiko(self, image: np.ndarray) -> torch.Tensor:
        
        DINO_HW = 1036
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(size=DINO_HW, antialias=True),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        image = transform(image).unsqueeze(0).to("cuda")
        
        return image
