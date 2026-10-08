from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

data_arr = [] # 데이터의 최소, 최댓값을 찾기 위한 배열 선언
data_dir = Path(os.getenv("T91_PATH"))
width_avg = 0
height_avg = 0
data_size = 0
for img_path in data_dir.iterdir():
    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        img = Image.open(img_path)
        data_arr.append(img.size)
        print(f"{img_path.name}: {img.size}")
        data_size+=1
        width_avg += img.size[0]
        height_avg += img.size[1]

width_avg /= data_size
height_avg /= data_size

print("Dataset Size : ", data_size)
print(f"Max Width : {max(data_arr[0])}\nMin Width : {min(data_arr[0])}\nMax Height : {max(data_arr[1])}\nMin Height : {min(data_arr[1])}\n")
print(f"Width Average : {width_avg: .2f}\nHeight Average : {height_avg: .2f}")
