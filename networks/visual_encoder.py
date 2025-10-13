import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """Simple CNN-based visual encoder producing a fixed-size feature vector."""

    def __init__(self, in_channels=3, base_channels=32, out_dim=512):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(4):
            layers.extend([
                nn.Conv2d(channels, base_channels * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * (2 ** i)),
                nn.ReLU(inplace=True)
            ])
            channels = base_channels * (2 ** i)
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, out_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        return self.fc(pooled)
