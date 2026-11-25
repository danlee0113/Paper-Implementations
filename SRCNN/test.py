
from pathlib import Path
import cv2
import numpy as np
import torch
from dotenv import load_dotenv
import os
import argparse

import wandb

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from model import SRCNN


load_dotenv()

# 데이터셋 이름 -> .env 경로 변수
DATASET_ENV = {
    "Set5": "SET5_PATH",
    "Set14": "SET14_PATH",
}


def parse_args():
    parser = argparse.ArgumentParser(description="SRCNN evaluation")
    parser.add_argument("--checkpoint", type=str, default="srcnn_v6_best.pth", help="평가할 모델 checkpoint 경로")
    parser.add_argument("--dataset", type=str, default="Set5", choices=list(DATASET_ENV), help="평가 데이터셋 (경로는 .env에서 읽음)")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--blur-sigma", type=float, default=1.0, help="Gaussian blur sigma (학습 때와 동일해야 함)")
    parser.add_argument("--blur-ksize", type=int, default=5, help="Gaussian blur kernel size (홀수)")
    parser.add_argument("--exp-name", type=str, default=None, help="wandb run name (미지정 시 '{checkpoint 이름}-{dataset}')")
    parser.add_argument("--wandb-project", type=str, default="SRCNN")
    parser.add_argument("--no-wandb", action="store_true", help="wandb 로깅 끄기")
    return parser.parse_args()


def to_image(tensor):
    """Convert normalized tensor to uint8 image."""
    return (
        tensor.detach()
        .clamp(0, 1)
        .squeeze()
        .cpu()
        .numpy() * 255
    ).astype(np.uint8)


def main():
    args = parse_args()

    scale = args.scale
    dataset = args.dataset
    checkpoint = Path(args.checkpoint)
    blur_ksize = (args.blur_ksize, args.blur_ksize)

    env_key = DATASET_ENV[dataset]
    if os.getenv(env_key) is None:
        raise ValueError(f"{env_key} is not set in .env")
    data_dir = Path(os.getenv(env_key))

    run_name = args.exp_name or f"{checkpoint.stem}-{dataset}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        job_type="evaluation",
        mode="disabled" if args.no_wandb else "online",
        config={
            **vars(args),
            "model": "SRCNN",
        }
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load trained model
    model = SRCNN().to(device)
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    psnr_scores = []
    ssim_scores = []

    bicubic_psnr_scores = []
    bicubic_ssim_scores = []

    image_paths = sorted(data_dir.glob("*.png"))

    if not image_paths:
        raise ValueError(f"No PNG images found in {dataset} directory: {data_dir}")

    # Store image comparison results
    comparison_images = []

    for image_path in image_paths:

        # Load HR image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)

        # BGR -> YCrCb -> Y
        y = cv2.cvtColor(
            image, cv2.COLOR_BGR2YCrCb
        )[:, :, 0]

        # Make dimensions divisible by scale
        h, w = y.shape
        h = (h // scale) * scale
        w = (w // scale) * scale
        hr = y[:h, :w]

        # Same degradation as training
        blurred = cv2.GaussianBlur(
            hr, blur_ksize, sigmaX=args.blur_sigma
        )

        # 블록 중앙 픽셀 샘플링 (cv2.resize 좌표 기준과 맞춤, preprocess.downsample과 동일)
        offset = scale // 2
        lr_small = blurred[offset::scale, offset::scale]

        lr = cv2.resize(
            lr_small,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        # Convert to tensor
        lr_tensor = torch.from_numpy(
            lr.astype(np.float32) / 255.0
        )[None, None].to(device)

        hr_tensor = torch.from_numpy(
            hr.astype(np.float32) / 255.0
        )[None, None].to(device)

        # SRCNN inference
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        # Crop HR and Bicubic to match SRCNN output
        # 33x33 -> 21x21
        bicubic_tensor = lr_tensor[:, :, 6:-6, 6:-6]
        hr_crop = hr_tensor[:, :, 6:-6, 6:-6]

        # Convert to NumPy
        sr_np = sr_tensor.squeeze().cpu().numpy()
        bicubic_np = bicubic_tensor.squeeze().cpu().numpy()
        hr_np = hr_crop.squeeze().cpu().numpy()

        # SRCNN evaluation
        psnr = peak_signal_noise_ratio(
            hr_np, sr_np, data_range=1.0
        )

        ssim = structural_similarity(
            hr_np, sr_np, data_range=1.0
        )

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

        # Bicubic evaluation
        bicubic_psnr = peak_signal_noise_ratio(
            hr_np, bicubic_np, data_range=1.0
        )

        bicubic_ssim = structural_similarity(
            hr_np, bicubic_np, data_range=1.0
        )

        bicubic_psnr_scores.append(bicubic_psnr)
        bicubic_ssim_scores.append(bicubic_ssim)

        print(
            f"{image_path.name}: "
            f"SRCNN PSNR={psnr:.2f}, "
            f"Bicubic PSNR={bicubic_psnr:.2f}"
        )

        # Log metrics for each image
        wandb.log({
            f"{dataset}/{image_path.stem}/srcnn_psnr": psnr,
            f"{dataset}/{image_path.stem}/srcnn_ssim": ssim,
            f"{dataset}/{image_path.stem}/bicubic_psnr": bicubic_psnr,
            f"{dataset}/{image_path.stem}/bicubic_ssim": bicubic_ssim
        })

        # Prepare visualization
        comparison = np.concatenate(
            [
                to_image(bicubic_tensor),
                to_image(sr_tensor),
                to_image(hr_crop)
            ],
            axis=1
        )

        comparison_images.append(
            wandb.Image(
                comparison,
                caption=(
                    f"{image_path.stem} | "
                    f"Left: Bicubic | "
                    f"Middle: SRCNN | "
                    f"Right: HR | "
                    f"PSNR: {psnr:.2f} dB"
                )
            )
        )

    # Calculate average metrics
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    avg_bicubic_psnr = np.mean(bicubic_psnr_scores)
    avg_bicubic_ssim = np.mean(bicubic_ssim_scores)

    psnr_improvement = avg_psnr - avg_bicubic_psnr

    print(f"\n===== {dataset} Results =====")

    print(
        f"SRCNN - PSNR: {avg_psnr:.2f} dB, "
        f"SSIM: {avg_ssim:.4f}"
    )

    print(
        f"Bicubic - PSNR: {avg_bicubic_psnr:.2f} dB, "
        f"SSIM: {avg_bicubic_ssim:.4f}"
    )

    print(
        f"PSNR Improvement: "
        f"{psnr_improvement:+.2f} dB"
    )

    # Log average metrics and images
    wandb.log({
        f"{dataset}/average/srcnn_psnr": avg_psnr,
        f"{dataset}/average/srcnn_ssim": avg_ssim,
        f"{dataset}/average/bicubic_psnr": avg_bicubic_psnr,
        f"{dataset}/average/bicubic_ssim": avg_bicubic_ssim,
        f"{dataset}/average/psnr_improvement": psnr_improvement,
        f"{dataset}/comparison_images": comparison_images
    })

    wandb.finish()


if __name__ == "__main__":
    main()
