"""
SRCNN 데이터 전처리 모듈.

파이프라인 (논문 기준):
    원본 RGB 이미지
      -> Y 채널 분리 (YCrCb)
      -> h, w를 scale의 배수로 자르기 (mod crop)          => GT 이미지 (X)
      -> Gaussian Blur (5x5, sigma=1.0)
      -> subsampling (factor = 1/scale)
      -> bicubic upsampling (factor = scale)            => 네트워크 입력 (Y)
      -> 33x33 patch로 crop (stride = 14)

사용법:
    python preprocess.py                      # 최종 patch만 저장 (CROP_PATH, GT_PATCH_PATH)
    python preprocess.py --save-intermediate  # 중간 결과도 저장 (Y_PATH, BLUR_PATH, LR_PATH, HR_PATH)

다른 파일에서 import 해서 사용할 수도 있음:
    from preprocess import rgb_to_y, mod_crop, degrade
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

SCALE = 3
PATCH_SIZE = 33
STRIDE = 14
BLUR_KSIZE = (5, 5)
BLUR_SIGMA = 1.0


# ---------------------------------------------------------------------------
# 단계별 함수
# ---------------------------------------------------------------------------
def rgb_to_y(image_bgr: np.ndarray) -> np.ndarray:
    """BGR 이미지 -> Y 채널 (YCrCb의 0번 채널)."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]


def mod_crop(image: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """h, w를 scale의 배수로 맞추기 (upsampling 후에도 크기가 같도록)."""
    h, w = image.shape[:2]
    return image[: (h // scale) * scale, : (w // scale) * scale]


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, ksize=BLUR_KSIZE, sigmaX=BLUR_SIGMA)


def downsample(image: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """factor 1/scale subsampling.

    각 scale x scale 블록의 중앙 픽셀(offset = scale // 2)을 샘플링.
    cv2.resize는 LR 픽셀 i를 HR 좌표 scale*i + (scale-1)/2 (블록 중앙)에 있다고 보고
    업샘플링하므로, [::scale]로 블록 왼쪽 끝을 뽑으면 결과가 1px 밀림.
    """
    offset = scale // 2
    return image[offset::scale, offset::scale]


def upsample(image: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """bicubic upsampling. cv2.resize는 (w, h) 순서임에 주의."""
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def degrade(hr_y: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """GT(Y 채널, mod crop 완료) -> 네트워크 입력 (blur -> down -> bicubic up).

    train/test 모두 이 함수를 쓰면 열화 과정이 항상 동일하게 유지됨.
    """
    return upsample(downsample(gaussian_blur(hr_y), scale), scale)


def extract_patches(image: np.ndarray, patch_size: int = PATCH_SIZE, stride: int = STRIDE):
    """이미지를 patch_size x patch_size patch로 나눔. (patch 하나씩 yield)"""
    h, w = image.shape[:2]
    for j in range(0, h - patch_size + 1, stride):      # height 축
        for k in range(0, w - patch_size + 1, stride):  # width 축
            yield image[j : j + patch_size, k : k + patch_size]


# ---------------------------------------------------------------------------
# 전체 파이프라인
# ---------------------------------------------------------------------------
def _env_dir(name: str) -> Path:
    value = os.getenv(name)
    if value is None:
        raise EnvironmentError(f"환경 변수 {name} 가 .env에 설정되어 있지 않습니다.")
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def preprocess(save_intermediate: bool = False) -> None:
    load_dotenv()

    data_dir = Path(os.getenv("T91_PATH"))
    crop_dir = _env_dir("CROP_PATH")      # 입력 patch (Y_i)
    gt_dir = _env_dir("GT_PATCH_PATH")    # GT patch (X_i)

    if save_intermediate:
        y_dir = _env_dir("Y_PATH")
        blur_dir = _env_dir("BLUR_PATH")
        lr_dir = _env_dir("LR_PATH")
        hr_dir = _env_dir("HR_PATH")

    img_paths = sorted(data_dir.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"{data_dir} 에 PNG 이미지가 없습니다.")

    total_patches = 0
    for i, img_path in enumerate(img_paths, start=1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load: {img_path}")
            continue

        # 1. Y 채널 분리 + mod crop -> GT
        gt = mod_crop(rgb_to_y(image))

        # 2. blur -> downsample -> bicubic upsample -> 입력
        blurred = gaussian_blur(gt)
        lr = downsample(blurred)
        inp = upsample(lr)
        assert inp.shape == gt.shape, (inp.shape, gt.shape)

        if save_intermediate:
            cv2.imwrite(str(y_dir / f"image_{i}.png"), gt)
            cv2.imwrite(str(blur_dir / f"image_{i}.png"), blurred)
            cv2.imwrite(str(lr_dir / f"image_{i}.png"), lr)
            cv2.imwrite(str(hr_dir / f"image_{i}.png"), inp)

        # 3. 33x33 patch로 crop (stride 14). 파일명은 dataset.py 형식에 맞춤.
        patch_idx = 0
        for inp_patch, gt_patch in zip(extract_patches(inp), extract_patches(gt)):
            cv2.imwrite(str(crop_dir / f"image{i}_patch{patch_idx}.png"), inp_patch)
            cv2.imwrite(str(gt_dir / f"image{i}_gt_patch{patch_idx}.png"), gt_patch)
            patch_idx += 1

        total_patches += patch_idx
        print(f"[{i}/{len(img_paths)}] {img_path.name}: {gt.shape} -> {patch_idx} patches")

    print(f"Preprocessing complete. Total patches: {total_patches}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRCNN data preprocessing")
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Y / blur / LR / upsampled 중간 이미지도 저장",
    )
    args = parser.parse_args()
    preprocess(save_intermediate=args.save_intermediate)
