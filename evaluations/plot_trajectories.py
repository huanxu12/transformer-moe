import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.eval_odom import _load_poses, _align_poses  # noqa: E402


def _parse_sequences(seq_text: str) -> List[str]:
    if not seq_text:
        return []
    sequences: List[str] = []
    for item in seq_text.split(','):
        token = item.strip()
        if not token:
            continue
        if token.isdigit():
            token = f"{int(token):02d}"
        sequences.append(token)
    return sequences


def _to_xyz(poses: List[np.ndarray]) -> np.ndarray:
    return np.array([pose[:3, 3] for pose in poses], dtype=np.float64)


def _plot_trajectory(gt_xyz: np.ndarray,
                     pred_xyz: np.ndarray,
                     seq: str,
                     output_dir: Path,
                     dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="Ground truth", linewidth=1.6, color="#1f77b4")
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 2], label="Prediction", linewidth=1.4, color="#ff7f0e")
    ax.set_title(f"KITTI Sequence {seq}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.axis("equal")
    ax.legend(loc="best")

    output_path = output_dir / f"{seq}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot predicted vs ground-truth KITTI trajectories.")
    parser.add_argument("--data_path", type=str, default=str(ROOT / "data"), help="Path to KITTI odometry data root")
    parser.add_argument("--pred_dir", type=str, default="results", help="Directory containing predicted KITTI-format trajectories")
    parser.add_argument("--sequences", type=str, default="09,10", help="Comma-separated sequence ids")
    parser.add_argument("--output_dir", type=str, default="results/plots", help="Directory to store trajectory plots")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    parser.add_argument("--no_align", action="store_true", help="Disable Sim(3) alignment before plotting")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = _parse_sequences(args.sequences)
    if not sequences:
        raise ValueError("No sequences specified for plotting")

    written = []
    for seq in sequences:
        gt_path = data_path / "poses" / f"{seq}.txt"
        pred_path = pred_dir / f"{seq}.txt"
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing ground-truth poses: {gt_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predicted trajectory: {pred_path}")

        gt_poses = _load_poses(gt_path)
        pred_poses = _load_poses(pred_path)
        length = min(len(gt_poses), len(pred_poses))
        gt_poses = gt_poses[:length]
        pred_poses = pred_poses[:length]

        if not args.no_align:
            pred_poses = _align_poses(pred_poses, gt_poses, allow_scale=True)

        gt_xyz = _to_xyz(gt_poses)
        pred_xyz = _to_xyz(pred_poses)
        output_path = _plot_trajectory(gt_xyz, pred_xyz, seq, output_dir, dpi=args.dpi)
        written.append((seq, output_path, length))

    print("Trajectory plots written:")
    for seq, path, length in written:
        print(f"  seq {seq}: {length} poses -> {path}")


if __name__ == "__main__":
    main()
