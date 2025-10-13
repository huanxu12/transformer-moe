import torch
import torch.nn as nn


class DummyVisualEncoder(nn.Module):
    """A lightweight visual encoder placeholder."""
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x):
        # x: B x 3 x H x W
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)
        return feat


class DummyIMUEncoder(nn.Module):
    """A lightweight IMU encoder placeholder."""
    def __init__(self, input_dim=6, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim)
        )

    def forward(self, imu_seq):
        # imu_seq: B x T x D
        if imu_seq.dim() == 2:
            imu_seq = imu_seq.unsqueeze(0)
        B, T, D = imu_seq.shape
        x = imu_seq.reshape(B * T, D)
        x = self.net(x)
        x = x.view(B, T, -1).mean(dim=1)
        return x
