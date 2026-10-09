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
y_dir = Path(os.getenv("Y_PATH"))
blur_dir = Path(os.getenv("BLUR_PATH"))

for i in range(1,92):
    image_path = y_dir / f"image_{i}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to load: {image_path}")
        continue
    blur_image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=1.0)
    save_path =  blur_dir / f"image_{i}.png"
    cv2.imwrite(str(save_path), blur_image)

print("Gaussian Blur Successful!")


