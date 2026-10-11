from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os

import cv2
import numpy as np

load_dotenv()

 # 이미지를 33x33의 patch들로 나누는 과정. stride = 14
hr_dir = Path(os.getenv("HR_PATH")) # Gaussian Blur + Y channel 분리 + downsample 후 upsample까지 완료한 이미지
y_dir = Path(os.getenv("Y_PATH")) # 원본 Y channel 분리된 이미지 
cropped_dir = Path(os.getenv("CROP_PATH"))
gt_dir = Path(os.getenv("GT_PATCH_PATH"))
patch_size = 33
stride = 14

for i in range(1,92) : 
    patch_idx=0
    hr_path = hr_dir / f"image{i}.png"
    y_path = y_dir/ f"image_{i}.png"
    y_image = cv2.imread(str(y_path), cv2.IMREAD_UNCHANGED) # 채널 하나로 하기 위해 
    hr_image = cv2.imread(str(hr_path), cv2.IMREAD_UNCHANGED)
    h,w = y_image.shape
    h = (h // 3) * 3
    w = (w // 3) * 3
    y_image = y_image[:h,:w] # 이 원본 이미지가 가공된 이미지의 pixel들과 같게 맞춰주는 과정

    for j in range(0,h-patch_size+1,stride): # height 축
        for k in range(0,w-patch_size+1,stride): # weight 축
            cropped_path = cropped_dir/f"image{i}_patch{patch_idx}.png"
            gt_path = gt_dir/f"image{i}_gt_patch{patch_idx}.png"
            cropped_patch = hr_image[j:j+patch_size, k:k+patch_size]
            gt_patch = y_image[j:j+patch_size, k:k+patch_size]
            
            cv2.imwrite(str(cropped_path), cropped_patch)
            cv2.imwrite(str(gt_path), gt_patch)
            patch_idx+=1

    