"""
dataset.py
-----------
Custom PyTorch dataset for SPADE fine-tuning with side-by-side image pairs.
"""

from pathlib import Path
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RandomJitter256:
    """Resize to (512,256) and randomly flip horizontally."""
    def __init__(self, jitter=True):
        self.jitter = jitter

    def __call__(self, img):
        img = img.resize((512, 256), Image.BICUBIC)
        if self.jitter and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


class SideBySideDataset(Dataset):
    """
    Dataset for images that contain concatenated pairs (input|target).
    Left half: target, right half: input.
    """

    def __init__(self, root, train=True):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.jpg"))
        self.transform = RandomJitter256(jitter=train)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)
        w, h = img.size
        w2 = w // 2
        left = img.crop((0, 0, w2, h)).resize((256, 256), Image.BICUBIC)
        right = img.crop((w2, 0, w, h)).resize((256, 256), Image.BICUBIC)
        return to_tensor_norm(right), to_tensor_norm(left), self.files[idx].name
