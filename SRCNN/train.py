from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os

import cv2

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

from model import SRCNN
from dataset import SRCNNDataset

load_dotenv()


"""
We use two types of datasets in this implementation. 

1. cropped patches : Y channel images -> gaussian blur -> downsampling by 1/3 
-> upsampling by 3 (bicubic) -> cropped with stride 14, with the size of 33x33.
2. GT patches : patches cropped from original images with stride 14, with the size of 33x33 (Y channel)


"""
EXPERIMENT_NAME = 'srcnn-baseline'

# Hyperparameters
epochs = 100
batch_size = 128
learning_rate = 1e-4

# wandb initialization
wandb.init(
    project="SRCNN",
    name=EXPERIMENT_NAME,
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "scale_factor": 3
    }
)

yi_dir = Path(os.getenv("CROP_PATH"))  # Y_i in the paper. (bicubic upsampled version, so size : 33x33)
xi_dir = Path(os.getenv("GT_PATCH_PATH")) # X_i in the paper. Ground truth patches. (also size : 33x33)





# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


train_dataset = SRCNNDataset(yi_dir, xi_dir)
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size, 
    shuffle = True,
    num_workers = 4
)

model = SRCNN().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr =learning_rate
)

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for lr, hr in train_loader:
        lr = lr.to(device)
        hr = hr.to(device)

        # forward pass
        sr = model(lr)

        # Crop HR: 33x33 -> 21x21
        hr = hr[:, :, 6:-6, 6:-6]
        loss = criterion(sr, hr)

        # backward pass
        optimizer.zero_grad() # 이전의 gradient 초기화
        loss.backward() # gradient 계산
        optimizer.step() # parameter 업데이트 


        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    wandb.log({
        "train/loss": avg_loss,
        "train/epoch": epoch + 1,
    })


    print(
        f"Epoch [{epoch+1}/{epochs}], "
        f"Loss: {avg_loss:.6f}"
    )
torch.save(model.state_dict(), "srcnn.pth")
wandb.finish()
