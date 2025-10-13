import torch
import torch.nn as nn


class Trans_Fusion(nn.Module):
    """Lightweight feature-level fusion for visual, IMU and point cloud encoders."""

    def __init__(self, dim, imu_dim=256, point_dim=None, drop_prob=0.0):
        super().__init__()
        self.dim = dim
        self.point_dim = point_dim

        # project modalities to visual feature space
        self.imu_proj = nn.Linear(imu_dim, dim)
        self.imu_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        if point_dim is not None:
            self.point_proj = nn.Linear(point_dim, dim)
            self.point_gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )
        else:
            self.point_proj = None
            self.point_gate = None

        self.dropout = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, visual_feat, imu_feat, point_feat=None, point_valid=None):
        # visual_feat, imu_feat : B x dim / imu_dim
        fused = visual_feat

        imu_proj = self.imu_proj(imu_feat)
        imu_input = torch.cat([fused, imu_proj], dim=1)
        imu_weight = self.imu_gate(imu_input)
        fused = fused + self.dropout(imu_weight * imu_proj)

        if self.point_proj is not None and point_feat is not None:
            if point_feat.dim() > 2:
                point_feat = point_feat.mean(dim=1)
            point_proj = self.point_proj(point_feat)
            gate_input = torch.cat([fused, point_proj], dim=1)
            point_weight = self.point_gate(gate_input)
            if point_valid is not None:
                if not isinstance(point_valid, torch.Tensor):
                    point_valid = torch.as_tensor(point_valid, dtype=point_weight.dtype, device=point_weight.device)
                point_weight = point_weight * point_valid.view(-1, 1)
            fused = fused + self.dropout(point_weight * point_proj)

        return fused
