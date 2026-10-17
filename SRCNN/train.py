from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os

import cv2

import torch 
import torch.nn as nn
import torch.optim as optim

from model import SRCNN


load_dotenv()


"""
We use two types of datasets in this implementation. 

1. cropped patches : Y channel images -> gaussian blur -> downsampling by 1/3 
-> upsampling by 3 (bicubic) -> cropped with stride 14, with the size of 33x33.
2. GT patches : patches cropped from original images with stride 14, with the size of 33x33 (Y channel)


"""

# Hyperparameters
epochs = 
learning_rate =
batch_size = 

# device 
# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


yi_dir = Path(os.getenv("CROP_PATH"))  # Y_i in the paper. (bicubic upsampled version, so size : 33x33)
xi_dir = Path(os.getenv("GT_PATCH_PATH")) # X_i in the paper. Ground truth patches. (also size : 33x33)

