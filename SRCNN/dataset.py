from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os

import cv2

import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


load_dotenv()


"""
We use two types of datasets in this implementation. 

1. cropped patches(lr images) : Y channel images -> gaussian blur -> downsampling by 1/3 
-> upsampling by 3 (bicubic) -> cropped with stride 14, with the size of 33x33.
2. GT patches(hr images) : patches cropped from original images with stride 14, with the size of 33x33 (Y channel)


"""

lr_dir = Path(os.getenv("CROP_PATH"))  # Y_i in the paper. (bicubic upsampled version, so size : 33x33)
hr_dir = Path(os.getenv("GT_PATCH_PATH")) # X_i in the paper. Ground truth patches. (also size : 33x33)


class SRCNNDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir

        self.lr_paths = sorted(
            self.lr_dir.glob("*.png"), # png로 끝나는 모든 이미지에 대해 접근
            key=lambda path: tuple(
                map(int, path.stem.replace("image", "").split("_patch"))
            )
        )

    def __len__(self):
        return len(self.lr_paths) # dataset의 크기 반환

    def __getitem__(self, idx):
        lr_path = self.lr_paths[idx] # dataset의 특정 데이터 가져오기
        hr_path = self.hr_dir / lr_path.name


        # Load images
        lr = np.array(Image.open(lr_path), dtype=np.float32)
        hr = np.array(Image.open(hr_path), dtype=np.float32)

        # Normalize [0, 255] -> [0, 1]
        lr = lr / 255.0 
        hr = hr / 255.0

        # Convert to Tensor: (C, H, W)
        lr = torch.from_numpy(lr).unsqueeze(0)
        hr = torch.from_numpy(hr).unsqueeze(0)

        return lr, hr
