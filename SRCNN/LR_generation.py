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

blur_dir = Path(os.getenv("BLUR_PATH"))
lr_dir = Path(os.getenv("LR_PATH"))# lr, but not cropped images(원본 이미지에서 subsampling(factor = 1/3)만 진행한 이미지)
for i in range(1,92) : 
    blur_image_path = blur_dir / f"image_{i}.png"
    blur_image = cv2.imread(str(blur_image_path), cv2.IMREAD_UNCHANGED) # 채널 하나로 하기 위해 
    if blur_image is None :
        print("Could not read image!\n")
        continue
    h,w = blur_image.shape
    h = (h // 3) * 3 # h,w 모두 3의 배수로 맞추기 (upsampling 후에도 크기가 맞도록)
    w = (w // 3) * 3
    blur_image=blur_image[:h,:w]
    lr_image = blur_image[::3,::3]
    print(f"Original shape : {blur_image.shape} , Downsampled shape : {lr_image.shape}\n")
    save_path = lr_dir / f"image{i}.png"
    cv2.imwrite(str(save_path),lr_image)
print("Downsampling is complete. The output is not cropped yet.")