import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import SRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRCNN().to(device)
model.load_state_dict(torch.load("srcnn.pth", map_location=device, weights_only=True))
model.eval()

@torch.inference_mode()
def evaluate(lr, hr):
    lr = lr.to(device)
    hr = hr.to(device)

    sr = model(lr)

    # Padding 없는 SRCNN의 출력 크기에 맞춰 HR crop
    hr = hr[:, :, 6:-6, 6:-6]

    assert sr.shape == hr.shape

    sr_np = sr.squeeze().cpu().numpy()
    hr_np = hr.squeeze().cpu().numpy()

    psnr = peak_signal_noise_ratio(
        hr_np, sr_np, data_range=1.0
    )

    ssim = structural_similarity(
        hr_np, sr_np, data_range=1.0
    )

    return psnr, ssim

    