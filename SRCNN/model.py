from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os

import cv2

import torch
import torch.nn as nn

load_dotenv()


"""
SRCNN is a 3-layer CNN.

- Architecture : Conv1 -> ReLU ->  Conv2 -> ReLU -> Conv3 
- Only uses Y channel.
- SGD
"""

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 9, padding = 0, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 1, padding = 0, stride = 1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 5, padding = 0, stride = 1)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x


    
