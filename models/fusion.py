import torch
import torch.nn as nn

class LearnableFusion(nn.Module):
    """Learnable fusion mechanism for combining spatial and frequency streams"""
    def __init__(self, spatial_dim, frequency_dim, fusion_dim=512):
        super(LearnableFusion, self).__init__()
        
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.frequency_proj = nn.Sequential(
            nn.Linear(frequency_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.attention = nn.Sequential(
            nn.Linear(2*fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, spatial_features, frequency_features):
        spatial_projection = self.spatial_proj(spatial_features)
        frequency_projection = self.frequency_proj(frequency_features)

        concat_features = torch.cat([spatial_projection, frequency_projection], dim=1) 

        attention_weights = self.attention(concat_features)

        fused = (attention_weights[:, 0:1]*spatial_features + attention_weights[:, 1:2]*frequency_features)

        return fused, attention_weights