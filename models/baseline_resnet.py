import torch
import torch.nn as nn
from torchvision import models

class ResNetBaseline(nn.Module):
    """
    Baseline ResNet18 model without CBAM or dual-stream.
    Used for ablation study comparison.
    
    Note: This is just a wrapper around torchvision's ResNet18 to match
    how it was trained in train_baseline_resnet.py
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNetBaseline, self).__init__()
        
        # Load ResNet18 (same as training script)
        base_model = models.resnet18(pretrained=pretrained)
        
        # Copy all layers except the final fc
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        
        # Custom fc layer for binary classification
        num_ftrs = base_model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x