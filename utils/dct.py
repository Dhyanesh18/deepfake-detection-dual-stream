import numpy as np
from scipy.fftpack import dct
import torch

def dct2d(block):
    """Apply 2D Discrete Cosince Transform (DCT) on an image channel"""
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

def rgb_to_dct_batch(images):
    """
    Convert a batch of RGB images to DCT domain.

    Args:
        images: torch.Tensor or np.ndarray of shape (B, 3, H, W)
                pixel values in [0, 255] or [0, 1]
    
    Returns:
        dct_images: torch.Tensor of shape (B, 3, H, W)
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    batch_size, _, H, W = images.shape
    
    dct_batch = []
    for b in range(batch_size):
        dct_channels = []
        for c in range(3):
            # Apply DCT to each color channel
            dct_ch = dct2d(images[b, c].astype(np.float32))
            dct_channels.append(dct_ch)
        dct_batch.append(np.stack(dct_channels, axis=0))
    
    dct_batch = np.stack(dct_batch, axis=0)  # (B, 3, H, W)
    return torch.from_numpy(dct_batch).float()


def normalize_dct(dct_batch):
    """Normalize DCT coefficients batch-wise."""
    mean = dct_batch.mean(dim=(2,3), keepdim=True)
    std = dct_batch.std(dim=(2,3), keepdim=True)
    return (dct_batch - mean) / (std + 1e-8)