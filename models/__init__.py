"""
Models package initialization
Exports all model architectures for deepfake detection
"""

from .backbones import XceptionWithCBAM, DCTStream, DualStreamModel
from .attention import CBAM, ChannelAttention, SpatialAttention
from .fusion import LearnableFusion

# Import baseline model (if it exists in baseline.py)
try:
    from .baseline import XceptionBaseline
except ImportError:
    # If XceptionBaseline doesn't exist, we'll create it below
    XceptionBaseline = None

__all__ = [
    'XceptionWithCBAM',
    'ResNet18WithCBAM',
    'DCTStream', 
    'DualStreamModelResNet',
    'DualStreamModel',
    'XceptionBaseline',
    'CBAM',
    'ChannelAttention',
    'SpatialAttention',
    'LearnableFusion'
]