from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


data_dir = Path(os.getenv("T91_PATH"))

data_size = 0
for img_path in data_dir.iterdir():
    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        img = Image.open(img_path)

        print(f"{img_path.name}: {img.size}")
        data_size+=1

print("Dataset Size : ", data_size)