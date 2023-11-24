from pathlib import Path
import torch
import torch.nn.functional as F


def preprocess_data(x, height, width, device):
    x = F.interpolate(x, size=(height, width), mode='nearest').float()  # (b c h w)
    x = x.to(device)
    # discrete -> continuous
    x = torch.argmax(x, dim=1, keepdim=True).float()  # (b 1 h w)
    return x
