import random
from pathlib import Path

import cv2

import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset

class OpenImageDatasetFPN(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = sorted([f for f in splitdir.iterdir() if f.is_file()])
        self.transform = transform
        self.mode = split

    def __getitem__(self, index):
        img = cv2.imread(str(self.samples[index]))
        if self.transform is not None:
            img, _ = T.apply_transform_gens(self.transform, img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        return img

    def __len__(self):
        return len(self.samples)
