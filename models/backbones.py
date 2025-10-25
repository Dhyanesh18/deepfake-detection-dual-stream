import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .fusion import LearnableFusion

from .attention import CBAM


class XceptionWithCBAM(nn.Module):
    """
    Xception backbone enhanced with CBAM modules at stages 3 and 4.

    This architecture integrates Convolutional Block Attention Modules (CBAM)
    into the mid- and high-level stages of the Xception model to improve
    spatial and channel attention. The goal is to enhance feature representation
    for fine-grained recognition tasks such as facial analysis.

    Stage Overview (for input ~300x300):
        - Stage 0: 64 channels  @ 150x150 (stride 2)  - Low-level features
        - Stage 1: 128 channels @ 75x75   (stride 4)  - Edges, textures
        - Stage 2: 256 channels @ 38x38   (stride 8)  - Local patterns
        - Stage 3: 728 channels @ 19x19   (stride 16) - Object parts [CBAM HERE]
        - Stage 4: 2048 channels @ 10x10  (stride 32) - High-level semantics [CBAM HERE]

    CBAM placement rationale:
        - Stage 3: Focuses on mid-level spatial details (e.g., eyes, nose, mouth)
        - Stage 4: Captures global semantic information (e.g., face composition)
        - This combination provides a good accuracy-speed tradeoff.

    Args:
        pretrained (bool, optional): If True, loads pretrained ImageNet weights
            for the Xception backbone. Defaults to True.
    """

    def __init__(self, pretrained: bool = True):
        super(XceptionWithCBAM, self).__init__()

        # Load pretrained Xception model from timm
        base_model = timm.create_model('xception', pretrained=pretrained, features_only=True)

        # Divide Xception backbone into 5 stages for modularity and clarity
        self.stage0 = nn.Sequential(*list(base_model.children())[0:2])
        self.stage1 = nn.Sequential(*list(base_model.children())[2:4])
        self.stage2 = nn.Sequential(*list(base_model.children())[4:6])
        self.stage3 = nn.Sequential(*list(base_model.children())[6:12])
        self.stage4 = nn.Sequential(*list(base_model.children())[12:])

        # Insert CBAM modules into mid- and high-level feature maps
        self.cbam_stage3 = CBAM(728)
        self.cbam_stage4 = CBAM(2048)
        self.feature_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Xception backbone with CBAM enhancements.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W),
                where H and W are typically 300x300.

        Returns:
            torch.Tensor: Global feature vector of shape (B, 2048),
                representing semantically enriched image descriptors.
        """
        # Early feature extraction
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)

        # Mid-level features with CBAM
        x = self.stage3(x)
        x = self.cbam_stage3(x)

        # High-level features with CBAM
        x = self.stage4(x)
        x = self.cbam_stage4(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

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
        self.cbam1 = CBAM(512)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(512)

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
            spatial_dim=self.spatial_stream.feature_dim,
            frequency_dim=self.frequency_stream.feature_dim,
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