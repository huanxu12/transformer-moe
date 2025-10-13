import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_poses(path: Path) -> List[np.ndarray]:
    poses: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            values = [float(item) for item in line.strip().split()]
            if len(values) != 12:
                raise ValueError(f"Unexpected line format in {path}: {line}")
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = np.reshape(values, (3, 4))
            poses.append(mat)
    if not poses:
        raise ValueError(f"No poses loaded from {path}")
    return poses


def _extract_xyz(poses: List[np.ndarray]) -> np.ndarray:
    return np.array([pose[:3, 3] for pose in poses], dtype=np.float64)


def _umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Tuple[np.ndarray, float, np.ndarray]:
    """Returns rotation, scale, translation aligning src to dst."""
    if src.shape != dst.shape:
        raise ValueError("Source and destination must share the same shape")
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst

    cov_matrix = (dst_centered.T @ src_centered) / src.shape[0]
    u, s, vh = np.linalg.svd(cov_matrix)
    d = np.linalg.det(u @ vh)
    reflection = np.eye(3)
    reflection[2, 2] = d
    rotation = u @ reflection @ vh

    if with_scale:
        var_src = np.sum(src_centered ** 2) / src.shape[0]
        scale = np.trace(np.diag(s) @ reflection) / var_src
    else:
        scale = 1.0

    translation = mean_dst - scale * rotation @ mean_src
    return rotation, scale, translation


def _align_poses(pred: List[np.ndarray], gt: List[np.ndarray], allow_scale: bool = True) -> List[np.ndarray]:
    pred_xyz = _extract_xyz(pred)
    gt_xyz = _extract_xyz(gt)
    rotation, scale, translation = _umeyama_alignment(pred_xyz, gt_xyz, with_scale=allow_scale)

    aligned: List[np.ndarray] = []
    for pose in pred:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation @ pose[:3, :3]
        transform[:3, 3] = (scale * rotation @ pose[:3, 3]) + translation
        aligned.append(transform)
    return aligned


def _rotation_error(rot: np.ndarray) -> float:
    trace = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(trace))


def _compute_metrics(gt: List[np.ndarray], pred: List[np.ndarray]) -> Dict[str, float]:
    translations_gt = _extract_xyz(gt)
    translations_pred = _extract_xyz(pred)
    diff = translations_gt - translations_pred
    ate_rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    ate_mean = float(np.mean(np.linalg.norm(diff, axis=1)))
    ate_median = float(np.median(np.linalg.norm(diff, axis=1)))

    rot_errors: List[float] = []
    trans_errors: List[float] = []
    for idx in range(len(gt) - 1):
        gt_rel = np.linalg.inv(gt[idx]) @ gt[idx + 1]
        pred_rel = np.linalg.inv(pred[idx]) @ pred[idx + 1]
        rel_error = np.linalg.inv(pred_rel) @ gt_rel
        rot_errors.append(_rotation_error(rel_error[:3, :3]))
        trans_errors.append(float(np.linalg.norm(rel_error[:3, 3])))

    rpe_rot_rmse = float(np.sqrt(np.mean(np.square(rot_errors)))) if rot_errors else 0.0
    rpe_trans_rmse = float(np.sqrt(np.mean(np.square(trans_errors)))) if trans_errors else 0.0

    return {
        "ate_rmse": ate_rmse,
        "ate_mean": ate_mean,
        "ate_median": ate_median,
        "rpe_rot_rmse": rpe_rot_rmse,
        "rpe_trans_rmse": rpe_trans_rmse,
    }


def evaluate_sequence(seq: str,
                      data_path: Path,
                      pred_dir: Path,
                      allow_scale: bool) -> Tuple[str, Dict[str, float]]:
    gt_path = data_path / "poses" / f"{seq}.txt"
    pred_path = pred_dir / f"{seq}.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground truth pose file: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predicted trajectory file: {pred_path}")

    gt_poses = _load_poses(gt_path)
    pred_poses = _load_poses(pred_path)

    length = min(len(gt_poses), len(pred_poses))
    if length < 2:
        raise ValueError(f"Sequence {seq} has insufficient poses (len={length})")
    gt_poses = gt_poses[:length]
    pred_poses = pred_poses[:length]

    aligned_pred = _align_poses(pred_poses, gt_poses, allow_scale=allow_scale)
    metrics = _compute_metrics(gt_poses, aligned_pred)
    metrics["num_poses"] = length
    return seq, metrics


def aggregate(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    total_poses = sum(item["num_poses"] for item in metrics.values())
    if total_poses == 0:
        return {}

    agg = {
        "ate_rmse": 0.0,
        "ate_mean": 0.0,
        "ate_median": 0.0,
        "rpe_rot_rmse": 0.0,
        "rpe_trans_rmse": 0.0,
    }
    for seq_metrics in metrics.values():
        weight = seq_metrics["num_poses"]
        for key in agg:
            agg[key] += seq_metrics[key] * weight
    for key in agg:
        agg[key] /= total_poses
    agg["num_poses"] = total_poses
    return agg


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trajectories with ATE/RPE metrics")
    parser.add_argument("--data_path", type=str, default=str(ROOT / "data"), help="Path to KITTI odometry data root")
    parser.add_argument("--pred_dir", type=str, default="results", help="Directory with predicted KITTI-format trajectories")
    parser.add_argument("--sequences", type=str, default="09,10", help="Comma separated list of sequence ids to evaluate")
    parser.add_argument("--allow_scale", action="store_true", help="Allow similarity (Sim(3)) alignment when computing ATE")
    parser.add_argument("--json_output", type=str, default=None, help="Optional JSON file to dump metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    pred_dir = Path(args.pred_dir)
    sequences = []
    for item in args.sequences.split(","):
        token = item.strip()
        if not token:
            continue
        if token.isdigit():
            token = f"{int(token):02d}"
        sequences.append(token)

    results: Dict[str, Dict[str, float]] = {}
    for seq in sequences:
        seq_id, metrics = evaluate_sequence(seq, data_path, pred_dir, allow_scale=args.allow_scale)
        results[seq_id] = metrics

    overall = aggregate(results)

    print("ATE/RPE evaluation results:")
    for seq in sequences:
        if seq not in results:
            continue
        metrics = results[seq]
        print(f"  Sequence {seq} ({metrics['num_poses']} poses):")
        print(f"    ATE RMSE      : {metrics['ate_rmse']:.4f} m")
        print(f"    ATE mean      : {metrics['ate_mean']:.4f} m")
        print(f"    ATE median    : {metrics['ate_median']:.4f} m")
        print(f"    RPE trans RMSE: {metrics['rpe_trans_rmse']:.4f} m")
        print(f"    RPE rot RMSE  : {math.degrees(metrics['rpe_rot_rmse']):.4f} deg")

    if overall:
        print("Overall (weighted by pose count):")
        print(f"    ATE RMSE      : {overall['ate_rmse']:.4f} m")
        print(f"    ATE mean      : {overall['ate_mean']:.4f} m")
        print(f"    ATE median    : {overall['ate_median']:.4f} m")
        print(f"    RPE trans RMSE: {overall['rpe_trans_rmse']:.4f} m")
        print(f"    RPE rot RMSE  : {math.degrees(overall['rpe_rot_rmse']):.4f} deg")

    if args.json_output:
        payload = {"sequences": results, "overall": overall}
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Metrics written to {args.json_output}")


if __name__ == "__main__":
    main()
