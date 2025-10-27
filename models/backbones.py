import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .fusion import LearnableFusion

from .attention import CBAM


class XceptionWithCBAM(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Use timm's feature extractor
        self.backbone = timm.create_model(
            'xception',
            pretrained=pretrained,
            features_only=True  # gives list of feature maps
        )

        # Feature channel dimensions from timm (for xception)
        feature_channels = self.backbone.feature_info.channels()
        # For xception → [64, 128, 256, 728, 2048]

        self.cbam_stage3 = CBAM(feature_channels[-2])  # 728
        self.cbam_stage4 = CBAM(feature_channels[-1])  # 2048

        self.feature_dim = feature_channels[-1]  # 2048

    def forward(self, x):
        # Get intermediate feature maps
        feats = self.backbone(x)
        x3 = feats[-2]  # 728 channels
        x4 = feats[-1]  # 2048 channels

        # Apply CBAM
        x3 = self.cbam_stage3(x3)
        x4 = self.cbam_stage4(x4)

        # Global average pooling → feature vector
        x = F.adaptive_avg_pool2d(x4, 1).view(x4.size(0), -1)
        return x


class DCTStream(nn.Module):
    """
    Lightweight convolutional stream designed for DCT (frequency-domain) features.

    This module extracts and refines hierarchical frequency-domain representations
    using sequential convolutional blocks combined with CBAM attention modules.
    It complements spatial-domain models such as XceptionWithCBAM for
    multi-stream architectures (e.g., DeepFake detection).

    Architecture:
        - conv1:  3 → 64   @ 150x150
        - conv2: 64 → 128  @ 75x75   [CBAM #1]
        - conv3: 128 → 256 @ 38x38   [CBAM #2]
        - conv4: 256 → 512 @ 19x19   [CBAM #3]

    Args:
        None
    """

    def __init__(self):
        super(DCTStream, self).__init__()

        # Convolutional feature extraction layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        )

        # CBAM attention modules for progressive refinement
        self.cbam1 = CBAM(128)  
        self.cbam2 = CBAM(256)  
        self.cbam3 = CBAM(512)  
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DCT stream.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W),
                typically representing DCT coefficient maps or frequency images.

        Returns:
            torch.Tensor: Global pooled feature vector of shape (B, 512),
                containing frequency-domain representations enhanced by attention.
        """
        # Hierarchical convolutional feature extraction
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.cbam1(x)

        x = self.conv3(x)
        x = self.cbam2(x)

        x = self.conv4(x)
        x = self.cbam3(x)

        # Global average pooling for compact feature vector
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        return x

class DualStreamModel(nn.Module):
    """Complete dual-stream model with spatial and frequency streams"""
    def __init__(self, num_classes=2, pretrained=True):
        super(DualStreamModel, self).__init__()
        self.spatial_stream = XceptionWithCBAM(pretrained=pretrained)
        self.frequency_stream = DCTStream()
        self.fusion = LearnableFusion(
            spatial_dim=2048,
            frequency_dim=512,
            fusion_dim=512
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, rgb_input, dct_input):
        spatial_features = self.spatial_stream(rgb_input)
        frequency_features = self.frequency_stream(dct_input)
        fused_features, attention_weights = self.fusion(spatial_features, frequency_features)
        logits = self.classifier(fused_features)
        return logits, attention_weights