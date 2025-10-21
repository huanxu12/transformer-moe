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

    def freeze_stages(self, num_stages):
        if num_stages <= 0:
            return
        total_stages = sum(1 for m in self.mlp if isinstance(m, nn.Linear))
        num_stages = min(num_stages, total_stages)
        stage_idx = 0
        module_idx = 0
        while stage_idx < num_stages and module_idx < len(self.mlp):
            linear = self.mlp[module_idx]
            linear.requires_grad_(False)
            module_idx += 1
            if module_idx < len(self.mlp) and isinstance(self.mlp[module_idx], nn.BatchNorm1d):
                bn = self.mlp[module_idx]
                bn.requires_grad_(False)
                bn.eval()
                module_idx += 1
            if module_idx < len(self.mlp):
                activation = self.mlp[module_idx]
                activation.requires_grad_(False)
                module_idx += 1
            stage_idx += 1

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
