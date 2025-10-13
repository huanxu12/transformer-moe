import torch
from torch import nn


class PointEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dims=(64, 128, 256), out_dim=256, use_bn=True, activation=nn.ReLU):
        super(PointEncoder, self).__init__()
        layers = []
        last_c = in_channels
        for dim in feature_dims:
            layers.append(nn.Linear(last_c, dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(activation())
            last_c = dim
        self.mlp = nn.ModuleList(layers)
        if last_c != out_dim:
            self.proj = nn.Linear(last_c, out_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, points):
        # points: B x N x C
        if not isinstance(points, torch.Tensor):
            points = torch.as_tensor(points, dtype=torch.float32)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        B, N, C = points.shape
        x = points
        x = x.reshape(B * N, C)
        for layer in self.mlp:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            else:
                x = layer(x)
        x = x.view(B, N, -1)
        x = x.max(dim=1)[0]
        x = self.proj(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
