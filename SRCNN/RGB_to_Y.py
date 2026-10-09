from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os

import cv2

load_dotenv()


data_dir = Path(os.getenv("T91_PATH"))
y_dir = Path(os.getenv("Y_PATH"))

img_paths = sorted(data_dir.glob("*.png"))

for i, img_path in enumerate(img_paths, start=1):
    # 이미지 읽기 (BGR)
    image = cv2.imread(str(img_path))

    # BGR → YCrCb 변환 후 Y 채널만 분리
    y_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[ :, :,0]

    # 파일명: image_1.png, image_2.png, ...
    save_path = y_dir / f"image_{i}.png"

    # 이미지 저장
    cv2.imwrite(str(save_path), y_image)

    print(f"Saved: {save_path}")


print("Successfully converted RGB to YCrCb and seperated Y channel!") # Note that the channels are YCrCb and not YCbCr. In an image viewer, the image might look odd.

