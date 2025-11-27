import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .fusion import LearnableFusion
from .attention import CBAM


class ResNet18WithCBAM(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            'resnet18',
            pretrained=pretrained,
            features_only=True
        )

        # ResNet18: [64, 128, 256, 512]
        feature_channels = self.backbone.feature_info.channels()

        self.cbam_stage3 = CBAM(feature_channels[-2])  # 256
        self.cbam_stage4 = CBAM(feature_channels[-1])  # 512

        self.feature_dim = feature_channels[-1]  # 512

    def forward(self, x):
        feats = self.backbone(x)
        x3 = feats[-2]
        x4 = feats[-1]

        x3 = self.cbam_stage3(x3)
        x4 = self.cbam_stage4(x4)

        x = F.adaptive_avg_pool2d(x4, 1).view(x4.size(0), -1)
        return x


class DCTStream(nn.Module):
    """Lightweight frequency stream with CBAM attention"""
   
    def __init__(self):
        super(DCTStream, self).__init__()

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
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # CBAM attention modules
        self.cbam1 = CBAM(128)  
        self.cbam2 = CBAM(256)  
        self.cbam3 = CBAM(512)  
       
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
       
        x = self.conv2(x)
        x = self.cbam1(x)
       
        x = self.conv3(x)
        x = self.cbam2(x)
       
        x = self.conv4(x)
        x = self.cbam3(x)
       
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return x


class DualStreamModelResNet(nn.Module):
    """Dual-stream model with ResNet18 spatial + DCT frequency streams"""
   
    def __init__(self, num_classes=2, pretrained=True):
        super(DualStreamModelResNet, self).__init__()
       
        self.spatial_stream = ResNet18WithCBAM(pretrained=pretrained)
        self.frequency_stream = DCTStream()
       
        # Balanced dimensions: both streams output 512
        self.fusion = LearnableFusion(
            spatial_dim=512,
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