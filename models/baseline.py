import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class XceptionBaseline(nn.Module):
    """
    Baseline Xception model without CBAM or dual-stream.
    Used for ablation study comparison.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(XceptionBaseline, self).__init__()
        
        # Load pretrained Xception
        self.backbone = timm.create_model('xception', pretrained=pretrained, num_classes=0)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits