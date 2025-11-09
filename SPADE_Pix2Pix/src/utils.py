"""
utils.py
---------
Utility functions: seeding, losses, perceptual metrics, FID.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torch_fidelity import calculate_metrics


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def d_hinge(real, fake): return (F.relu(1. - real).mean() + F.relu(1. + fake).mean())
def g_hinge(fake): return -fake.mean()

l1 = nn.L1Loss()

def build_vgg(device):
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features[:16].eval().to(device)
    for p in vgg.parameters():
        p.requires_grad = False
    return vgg


def perceptual(x, y, vgg):
    return l1(vgg(x), vgg(y))


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def compute_fid(gt_dir, pred_dir):
    m = calculate_metrics(
        input1=str(gt_dir),
        input2=str(pred_dir),
        cuda=torch.cuda.is_available(),
        isc=False, fid=True, kid=False, verbose=False
    )
    return float(m['frechet_inception_distance'])
