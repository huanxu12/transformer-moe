import torch
import torch.nn as nn


class PoseRegressor(nn.Module):
    """Map fused features to 6-DoF pose parameters (axis-angle + translation)."""

    def __init__(self, in_dim=512, hidden_dim=256, dropout=0.0):
        super().__init__()
        dropout = float(max(0.0, min(1.0, dropout)))
        dropout_layer = nn.Dropout(dropout)
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            dropout_layer,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6)
        ]
        self.net = nn.Sequential(*layers)
        self._dropout_layer = dropout_layer

    def set_dropout(self, prob: float):
        prob = float(max(0.0, min(1.0, prob)))
        self._dropout_layer.p = prob

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pose = self.net(x)
        rot = pose[:, :3]
        trans = pose[:, 3:]
        return rot, trans
