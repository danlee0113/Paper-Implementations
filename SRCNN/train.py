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
EXPERIMENT_NAME = 'srcnn-baseline-v1'

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 128
LR_12 = 1e-4 # learning rate of layer 1,2 
LR_3 = 1e-5 # learning rate of layer 3
SCALE = 3

# wandb initialization
wandb.init(
    project="SRCNN",
    name=EXPERIMENT_NAME,
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate_12": LR_12,
        "learning_rate_3": LR_3,
        "optimizer": "SGD",
        "scale_factor": SCALE
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
    batch_size = BATCH_SIZE, 
    shuffle = True,
    num_workers = 4
)

model = SRCNN().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD([
    {"params": model.conv1.parameters()},                # lr을 안 적으면 아래 기본값 사용
    {"params": model.conv2.parameters()},
    {"params": model.conv3.parameters(), "lr": LR_3},    # 이 그룹만 따로 지정
], lr=LR_12, momentum =0.9
)

for epoch in range(EPOCHS):
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
        f"Epoch [{epoch+1}/{EPOCHS}], "
        f"Loss: {avg_loss:.6f}"
    )
torch.save(model.state_dict(), "srcnn_v1.pth")
wandb.finish()
