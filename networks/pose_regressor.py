import torch
import torch.nn as nn


class PoseRegressor(nn.Module):
    """Map fused features to 6-DoF pose parameters (axis-angle + translation)."""

    def __init__(self, in_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pose = self.net(x)
        rot = pose[:, :3]
        trans = pose[:, 3:]
        return rot, trans
