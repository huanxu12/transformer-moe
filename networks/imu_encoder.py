import torch
import torch.nn as nn


class IMUEncoder(nn.Module):
    """GRU-based IMU encoder producing a fixed feature vector."""

    def __init__(self, input_dim=6, hidden_dim=128, num_layers=1, out_dim=256, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, imu_seq):
        if not isinstance(imu_seq, torch.Tensor):
            imu_seq = torch.as_tensor(imu_seq, dtype=torch.float32)
        if imu_seq.dim() == 2:
            imu_seq = imu_seq.unsqueeze(0)
        output, hidden = self.gru(imu_seq)
        feat = hidden[-1]
        return self.fc(feat)
