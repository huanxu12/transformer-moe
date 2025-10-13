import os
import numpy as np
import torch

from .kitti_dataset import KITTIOdomDataset
from util import pointcloud_ops


class KITTIOdomPointDataset(KITTIOdomDataset):
    """KITTI odometry dataset with point cloud support."""

    def __init__(self, *args,
                 pointcloud_path=None,
                 pc_max_points=8192,
                 pc_min_range=0.5,
                 pc_max_range=80.0,
                 return_intensity=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pointcloud_root = pointcloud_path or os.path.join(self.data_path, "pointclouds")
        self.pc_max_points = pc_max_points
        self.pc_min_range = pc_min_range
        self.pc_max_range = pc_max_range
        self.return_intensity = return_intensity

    def _pointcloud_path(self, folder, frame_index):
        seq = f"{int(folder):02d}"
        filename = f"{frame_index:06d}.bin"
        return os.path.join(self.pointcloud_root, seq, filename)

    def _load_pointcloud(self, folder, frame_index):
        path = self._pointcloud_path(folder, frame_index)
        if not os.path.isfile(path):
            return None, 0
        points = pointcloud_ops.load_kitti_bin(path)
        points = pointcloud_ops.filter_by_range(points, self.pc_min_range, self.pc_max_range)
        valid_count = points.shape[0]
        if self.pc_max_points and valid_count > self.pc_max_points:
            points = pointcloud_ops.random_sample(points, self.pc_max_points)
            valid_count = points.shape[0]
        if not self.return_intensity and points.shape[1] == 4:
            points = points[:, :3]
        if self.pc_max_points and points.shape[0] < self.pc_max_points:
            pad = np.zeros((self.pc_max_points - points.shape[0], points.shape[1]), dtype=points.dtype)
            points = np.concatenate([points, pad], axis=0)
        return points, valid_count

    def __getitem__(self, index):
        inputs = super().__getitem__(index)
        line = inputs['filename'].split()
        folder = line[0]
        frame_index = int(line[1]) if len(line) >= 2 else 0

        points, valid_count = self._load_pointcloud(folder, frame_index)
        if points is None:
            inputs['pc_valid'] = torch.tensor(False)
            num_channels = 4 if self.return_intensity else 3
            max_points = self.pc_max_points or 0
            tensor = torch.zeros((max_points, num_channels), dtype=torch.float32)
            inputs[("pointcloud", 0, 0)] = tensor
            inputs['pc_count'] = torch.tensor(0)
        else:
            tensor = torch.from_numpy(points).float()
            inputs[("pointcloud", 0, 0)] = tensor
            inputs['pc_valid'] = torch.tensor(valid_count > 0)
            inputs['pc_count'] = torch.tensor(valid_count)
        return inputs
