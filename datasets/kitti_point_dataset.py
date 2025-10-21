import os
import math
import random
import numpy as np
import torch

from .kitti_dataset import KITTIOdomDataset
from util import pointcloud_ops


class KITTIOdomPointDataset(KITTIOdomDataset):
    """KITTI odometry dataset with point cloud support and optional augmentations."""

    def __init__(self, *args,
                 pointcloud_path=None,
                 pc_max_points=8192,
                 pc_min_range=0.5,
                 pc_max_range=80.0,
                 return_intensity=True,
                 lidar_sector_dropout_prob=0.0,
                 lidar_sector_dropout_angle=35.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pointcloud_root = pointcloud_path or os.path.join(self.data_path, "pointclouds")
        self.pc_max_points = pc_max_points
        self.pc_min_range = pc_min_range
        self.pc_max_range = pc_max_range
        self.return_intensity = return_intensity
        self.lidar_sector_dropout_prob = float(max(0.0, min(1.0, lidar_sector_dropout_prob)))
        self.lidar_sector_dropout_angle = float(max(0.0, lidar_sector_dropout_angle))

    def _pointcloud_path(self, folder, frame_index):
        seq = f"{int(folder):02d}"
        filename = f"{frame_index:06d}.bin"
        return os.path.join(self.pointcloud_root, seq, filename)

    def _apply_lidar_sector_dropout(self, points):
        if points.size == 0:
            return points
        sector_width = math.radians(self.lidar_sector_dropout_angle)
        if sector_width <= 0:
            return points
        yaw = np.arctan2(points[:, 1], points[:, 0])
        center = random.uniform(-math.pi, math.pi)
        diff = (yaw - center + math.pi) % (2 * math.pi) - math.pi
        mask = np.abs(diff) > sector_width * 0.5
        if mask.sum() == 0:
            return points
        return points[mask]

    def _load_pointcloud(self, folder, frame_index):
        path = self._pointcloud_path(folder, frame_index)
        if not os.path.isfile(path):
            return None, 0
        points = pointcloud_ops.load_kitti_bin(path)
        if self.is_train and self.lidar_sector_dropout_prob > 0.0:
            if random.random() < self.lidar_sector_dropout_prob:
                points = self._apply_lidar_sector_dropout(points)
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
