import numpy as np
import torch


def load_kitti_bin(path):
    """Load KITTI-format point cloud from .bin path."""
    data = np.fromfile(path, dtype=np.float32)
    if data.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    points = data.reshape(-1, 4)
    return points


def filter_by_range(points, min_range=0.0, max_range=80.0):
    if points.size == 0:
        return points
    coords = points[:, :3]
    dist = np.linalg.norm(coords, axis=1)
    mask = np.ones(dist.shape, dtype=bool)
    if min_range is not None:
        mask &= dist >= min_range
    if max_range is not None:
        mask &= dist <= max_range
    return points[mask]


def random_sample(points, max_points):
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = np.random.choice(points.shape[0], max_points, replace=False)
    return points[idx]


def to_tensor(points):
    if isinstance(points, torch.Tensor):
        return points.float()
    return torch.from_numpy(points).float()
