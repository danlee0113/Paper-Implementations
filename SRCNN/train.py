from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os
import argparse
import random

import numpy as np

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
def parse_args():
    parser = argparse.ArgumentParser(description="SRCNN training")
    # Experiment
    parser.add_argument("--exp-name", type=str, default="srcnn-baseline-v6", help="wandb run name, checkpoint 파일명 prefix")
    parser.add_argument("--seed", type=int, default=None, help="random seed (미지정 시 고정하지 않음)")
    # Hyperparameters
    parser.add_argument("--max-epochs", type=int, default=3000, help="early stopping이 안 걸릴 때의 상한")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-12", type=float, default=1e-4, help="learning rate of layer 1,2")
    parser.add_argument("--lr-3", type=float, default=1e-5, help="learning rate of layer 3")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    # Early stopping
    parser.add_argument("--patience", type=int, default=50, help="이 epoch 수 동안 개선이 없으면 종료")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="best 대비 상대 개선율 (0.01%% ≈ 0.0004 dB)")
    # Checkpoint
    parser.add_argument("--ckpt-dir", type=str, default=".", help="checkpoint 저장 디렉토리")
    # wandb
    parser.add_argument("--wandb-project", type=str, default="SRCNN")
    parser.add_argument("--no-wandb", action="store_true", help="wandb 로깅 끄기")
    return parser.parse_args()


args = parse_args()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

ckpt_dir = Path(args.ckpt_dir)
ckpt_dir.mkdir(parents=True, exist_ok=True)
BEST_CKPT = ckpt_dir / f"{args.exp_name}_best.pth"
LAST_CKPT = ckpt_dir / f"{args.exp_name}_last.pth"

# wandb initialization
wandb.init(
    project=args.wandb_project,
    name=args.exp_name,
    mode="disabled" if args.no_wandb else "online",
    config={
        **vars(args),
        "optimizer": "SGD",
        "loss": "sum over pixels / batch (paper eq.2)",
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
    batch_size = args.batch_size, 
    shuffle = True,
    num_workers = args.num_workers
)

model = SRCNN().to(device)

# 논문 식 (2): L = 1/n * sum_i ||F(Y_i) - X_i||^2
#   ||.||^2 는 patch 내 모든 픽셀의 제곱 오차 '합' -> 픽셀 수로는 나누지 않고 batch 크기로만 나눔.
#   (nn.MSELoss() 기본값(mean)은 21x21=441 픽셀로도 나눠서 gradient가 논문보다 441배 작아짐)
criterion = nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD([
    {"params": model.conv1.parameters()},                # lr을 안 적으면 아래 기본값 사용
    {"params": model.conv2.parameters()},
    {"params": model.conv3.parameters(), "lr": args.lr_3},    # 이 그룹만 따로 지정
], lr=args.lr_12, momentum=args.momentum
)

best_loss = float("inf")
best_epoch = 0
epochs_no_improve = 0

for epoch in range(args.max_epochs):
    model.train()
    running_loss = 0

    for lr, hr in train_loader:
        lr = lr.to(device)
        hr = hr.to(device)

        # forward pass
        sr = model(lr)

        # Crop HR: 33x33 -> 21x21
        hr = hr[:, :, 6:-6, 6:-6]
        loss = criterion(sr, hr) / sr.size(0)  # patch별 픽셀 합 -> batch 평균

        # backward pass
        optimizer.zero_grad() # 이전의 gradient 초기화
        loss.backward() # gradient 계산
        optimizer.step() # parameter 업데이트 


        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # Early stopping: best 대비 MIN_DELTA(상대값) 이상 줄었을 때만 개선으로 인정
    if avg_loss < best_loss * (1 - args.min_delta):
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
        f"Epoch [{epoch+1}/{args.max_epochs}], "
        f"Loss: {avg_loss:.6f}, "
        f"Best: {best_loss:.6f} (epoch {best_epoch}), "
        f"No improve: {epochs_no_improve}/{args.patience}"
    )

    if epochs_no_improve >= args.patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

torch.save(model.state_dict(), LAST_CKPT)

wandb.run.summary["best_loss"] = best_loss
wandb.run.summary["best_epoch"] = best_epoch
wandb.run.summary["stopped_epoch"] = epoch + 1
wandb.finish()
