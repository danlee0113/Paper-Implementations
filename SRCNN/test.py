
from pathlib import Path
import cv2
import numpy as np
import torch
from dotenv import load_dotenv
import os

import wandb

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from model import SRCNN


load_dotenv()

SCALE = 3
SET5_DIR = Path(os.getenv("SET5_PATH"))
SET14_DIR = Path(os.getenv("SET14_PATH"))
CHECKPOINT = Path("srcnn_v2.pth")
BLUR_SIGMA = 1.0
BLUR_KSIZE = (5, 5)

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
    wandb.init(
        project="SRCNN",
        name="srcnn-baseline-v2-Set14",
        job_type="evaluation",
        config={
            "model": "SRCNN",
            "dataset": "Set14",
            "scale_factor": SCALE
        }
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load trained model
    model = SRCNN().to(device)
    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    psnr_scores = []
    ssim_scores = []

    bicubic_psnr_scores = []
    bicubic_ssim_scores = []

    image_paths = sorted(SET14_DIR.glob("*.png"))

    if not image_paths:
        raise ValueError("No PNG images found in Set5 directory")

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

        # Make dimensions divisible by SCALE
        h, w = y.shape
        h = (h // SCALE) * SCALE
        w = (w // SCALE) * SCALE
        hr = y[:h, :w]

        # Same degradation as training
        blurred = cv2.GaussianBlur(
            hr, BLUR_KSIZE, sigmaX=BLUR_SIGMA
        )

        lr_small = blurred[::SCALE, ::SCALE]

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
            f"Set14/{image_path.stem}/srcnn_psnr": psnr,
            f"Set14/{image_path.stem}/srcnn_ssim": ssim,
            f"Set14/{image_path.stem}/bicubic_psnr": bicubic_psnr,
            f"Set14/{image_path.stem}/bicubic_ssim": bicubic_ssim
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

    print("\n===== Set14 Results =====")

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
        "Set14/average/srcnn_psnr": avg_psnr,
        "Set14/average/srcnn_ssim": avg_ssim,
        "Set14/average/bicubic_psnr": avg_bicubic_psnr,
        "Set14/average/bicubic_ssim": avg_bicubic_ssim,
        "Set14/average/psnr_improvement": psnr_improvement,
        "Set14/comparison_images": comparison_images
    })

    wandb.finish()


if __name__ == "__main__":
    main()
