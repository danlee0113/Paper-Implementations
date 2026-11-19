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
EXPERIMENT_NAME = 'srcnn-baseline-v3'

# Hyperparameters
MAX_EPOCHS = 1500 # early stopping이 안 걸릴 때의 상한
BATCH_SIZE = 128
LR_12 = 1e-4 # learning rate of layer 1,2
LR_3 = 1e-5 # learning rate of layer 3
SCALE = 3

# Early stopping
PATIENCE = 50     # 이 epoch 수 동안 개선이 없으면 종료
MIN_DELTA = 1e-4  # best 대비 상대 개선율 (0.01% ≈ 0.0004 dB)
BEST_CKPT = "srcnn_v3_best.pth"
LAST_CKPT = "srcnn_v3_last.pth"

# wandb initialization
wandb.init(
    project="SRCNN",
    name=EXPERIMENT_NAME,
    config={
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
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

best_loss = float("inf")
best_epoch = 0
epochs_no_improve = 0

for epoch in range(MAX_EPOCHS):
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

    # Early stopping: best 대비 MIN_DELTA(상대값) 이상 줄었을 때만 개선으로 인정
    if avg_loss < best_loss * (1 - MIN_DELTA):
        best_loss = avg_loss
        best_epoch = epoch + 1
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_CKPT)
    else:
        epochs_no_improve += 1

    wandb.log({
        "train/loss": avg_loss,
        "train/best_loss": best_loss,
        "train/epochs_no_improve": epochs_no_improve,
        "train/epoch": epoch + 1,
    })


    print(
        f"Epoch [{epoch+1}/{MAX_EPOCHS}], "
        f"Loss: {avg_loss:.6f}, "
        f"Best: {best_loss:.6f} (epoch {best_epoch}), "
        f"No improve: {epochs_no_improve}/{PATIENCE}"
    )

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

torch.save(model.state_dict(), LAST_CKPT)

wandb.run.summary["best_loss"] = best_loss
wandb.run.summary["best_epoch"] = best_epoch
wandb.run.summary["stopped_epoch"] = epoch + 1
wandb.finish()
