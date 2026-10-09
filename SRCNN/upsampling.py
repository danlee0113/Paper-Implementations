from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os

import cv2

load_dotenv()

"""
GT image X : random sampling from original image (33x33x1) -> Y channel only.
LR image Y : gaussian blur + subsampling with factor = 1/3 (11x11x1)
HR image Y : preprocessed image that goes into network. upsampling factor = 3(33x33x1)  
"""

hr_dir = Path(os.getenv("HR_PATH"))
lr_dir = Path(os.getenv("LR_PATH"))# lr, but not cropped images(원본 이미지에서 subsampling(factor = 1/3)만 진행한 이미지)
for i in range(1,92) : 
    lr_image_path = lr_dir / f"image{i}.png"
    lr_image = cv2.imread(str(lr_image_path), cv2.IMREAD_UNCHANGED) # 채널 하나로 하기 위해 
    if lr_image is None :
        print("Could not read image!\n")
        continue
    h,w = lr_image.shape # shape의 순서가 h,w임.
    hr_image = cv2.resize(lr_image, (w * 3, h * 3), interpolation = cv2.INTER_CUBIC) # bicubic interpolation은 w,h 순서임.
    print(f"Original shape : {lr_image.shape} , Upsampled shape : {hr_image.shape}\n")
    save_path = hr_dir / f"image{i}.png"
    cv2.imwrite(str(save_path),hr_image)
print("Upsampling is complete. The output is not cropped yet.")
