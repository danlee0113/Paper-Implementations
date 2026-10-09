from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os

import cv2

load_dotenv()

hr_dir = Path(os.getenv("HR_PATH"))
y_dir = Path(os.getenv("Y_PATH")) # 원본 Y 
for i in range(1,92) : 
    hr_path = hr_dir / f"image{i}.png"
    y_path = y_dir/ f"image_{i}.png"
    y_image = cv2.imread(str(y_path), cv2.IMREAD_UNCHANGED) # 채널 하나로 하기 위해 
    hr_image = cv2.imread(str(hr_path), cv2.IMREAD_UNCHANGED)

    print(f"Ground Truth Y : {y_image.shape}, Upsampled Y : {hr_image.shape}")
    assert y_image.shape == hr_image.shape, f"Shape mismatch: {y_image.shape} != {hr_image.shape}"