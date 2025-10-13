import os
import numpy as np


def _read_calib_file(path):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if not value:
                continue
            data[key] = np.array([float(x) for x in value.split()])
    return data


def _get_calib_dict(calib_path):
    if os.path.isdir(calib_path):
        txt_path = os.path.join(calib_path, 'calib.txt')
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Calibration file not found in {calib_path}")
        return _read_calib_file(txt_path)
    if os.path.isfile(calib_path):
        return _read_calib_file(calib_path)
    raise FileNotFoundError(f"Calibration path {calib_path} not found")


def _load_velodyne_points(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    points = points[points[:, 0] >= 0]
    return points


def generate_depth_map(calib_path, velo_filename, cam):
    calib = _get_calib_dict(calib_path)
    cam = int(cam)
    proj_key = f'P{cam}'
    if proj_key not in calib:
        raise KeyError(f"Projection matrix {proj_key} not found in calibration file")
    P = calib[proj_key].reshape(3, 4)

    if 'R0_rect' in calib:
        R_rect = calib['R0_rect'].reshape(3, 3)
    elif 'R_rect_00' in calib:
        R_rect = calib['R_rect_00'].reshape(3, 3)
    else:
        R_rect = np.eye(3)

    if 'Tr_velo_to_cam' in calib:
        Tr = calib['Tr_velo_to_cam'].reshape(3, 4)
    elif 'Tr' in calib:
        Tr = calib['Tr'].reshape(3, 4)
    else:
        raise KeyError("Tr_velo_to_cam matrix not found; please download KITTI odometry calibration files")

    Tr = np.vstack([Tr, [0, 0, 0, 1]])
    R_rect_4x4 = np.eye(4)
    R_rect_4x4[:3, :3] = R_rect
    proj = P @ R_rect_4x4 @ Tr

    pts = _load_velodyne_points(velo_filename)
    pts_hom = np.hstack((pts[:, :3], np.ones((pts.shape[0], 1))))
    pts_cam = proj @ pts_hom.T

    pts_cam[0, :] /= pts_cam[2, :]
    pts_cam[1, :] /= pts_cam[2, :]

    depth_map = np.zeros((375, 1242), dtype=np.float32)
    u = np.round(pts_cam[0, :]).astype(int)
    v = np.round(pts_cam[1, :]).astype(int)
    z = pts_cam[2, :]

    mask = (z > 0) & (u >= 0) & (u < depth_map.shape[1]) & (v >= 0) & (v < depth_map.shape[0])
    u = u[mask]
    v = v[mask]
    z = z[mask]

    depth_map[v, u] = np.where(depth_map[v, u] > 0, np.minimum(depth_map[v, u], z), z)
    return depth_map
