import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path('.').resolve()
GT_PATH = ROOT / 'data' / 'poses' / '09.txt'
BASELINE_PATH = ROOT / 'results' / 'stage3_baseline_traj' / '09.txt'
DROP_DIRS = {
    'stage3_pose_drop025': ROOT / 'results' / 'stage3_pose_drop025_traj' / '09.txt',
    'stage3_pose_drop030': ROOT / 'results' / 'stage3_pose_drop030_traj' / '09.txt',
    'stage3_pose_drop035': ROOT / 'results' / 'stage3_pose_drop035_traj' / '09.txt',
    'stage3_pose_dropout_schedule': ROOT / 'results' / 'stage3_pose_dropout_schedule' / '09.txt',
}
FRAME_MARKERS = {
    'stage3_pose_drop025': {
        1: [1538],
    },
    'stage3_pose_drop030': {
        1: [707, 755],
        2: [930, 954],
    },
    'stage3_pose_drop035': {
        2: [679],
    },
    'stage3_pose_dropout_schedule': {
        # add specific frames if needed
    },
}
OUTPUT_ROOT = ROOT / 'results' / 'analysis'
PLOT_ROOT = ROOT / 'results' / 'plots_drop_segments'
WINDOW_SIZE = 120  # roughly 12s at 10Hz
TOP_K = 3

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_ROOT.mkdir(parents=True, exist_ok=True)


def load_traj(path: Path) -> np.ndarray:
    data = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 12:
                continue
            mat = np.array(list(map(float, parts)), dtype=np.float64).reshape(3, 4)
            t = mat[:, 3]
            data.append(t)
    if not data:
        raise ValueError(f'No valid trajectory data found in {path}')
    return np.vstack(data)


def sliding_windows(errors: np.ndarray, window: int) -> np.ndarray:
    if len(errors) < window:
        return np.array([])
    cumsum = np.cumsum(np.insert(errors, 0, 0.0))
    window_sum = cumsum[window:] - cumsum[:-window]
    return window_sum / window


def pick_top_segments(mean_errors: np.ndarray, window: int, topk: int) -> List[int]:
    selected = []
    scores = mean_errors.copy()
    for _ in range(topk):
        if scores.size == 0:
            break
        idx = int(np.argmax(scores))
        if not np.isfinite(scores[idx]) or scores[idx] <= 0:
            break
        selected.append(idx)
        low = max(0, idx - window)
        high = min(scores.size, idx + window)
        scores[low:high] = -np.inf
    return selected


def annotate_point(ax, x: float, y: float, label: str, color: str):
    ax.scatter(x, y, color=color, marker='*', s=70, zorder=5)
    ax.text(x, y, label, color=color, fontsize=8, weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, pad=1.5))


def plot_segment(name: str, start_idx: int, window: int, gt: np.ndarray, baseline: np.ndarray, pred: np.ndarray, out_path: Path, marker_frames: List[int]):
    end_idx = min(start_idx + window, len(gt))
    frame_start = start_idx
    frame_end = end_idx - 1

    gt_xy = gt[start_idx:end_idx, :2]
    base_xy = baseline[start_idx:end_idx, :2]
    pred_xy = pred[start_idx:end_idx, :2]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], label='GT', color='black', linewidth=2)
    ax.plot(base_xy[:, 0], base_xy[:, 1], label='Baseline drop003', color='tab:blue')
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], label=name, color='tab:red')

    annotate_point(ax, gt_xy[0, 0], gt_xy[0, 1], f'Start {frame_start}', 'green')
    annotate_point(ax, gt_xy[-1, 0], gt_xy[-1, 1], f'End {frame_end}', 'purple')

    for frame in marker_frames:
        if frame_start <= frame <= frame_end:
            offset = frame - frame_start
            x, y = gt_xy[offset, 0], gt_xy[offset, 1]
            annotate_point(ax, x, y, f'{frame}', 'goldenrod')

    ax.set_title(f'{name} vs GT Segment [{frame_start}, {frame_end}]')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    gt = load_traj(GT_PATH)
    baseline = load_traj(BASELINE_PATH)
    results: Dict[str, Dict] = {}

    for name, path in DROP_DIRS.items():
        pred = load_traj(path)
        length = min(len(gt), len(pred), len(baseline))
        gt_aligned = gt[:length]
        baseline_aligned = baseline[:length]
        pred_aligned = pred[:length]

        errors_pred = np.linalg.norm(pred_aligned - gt_aligned, axis=1)
        errors_base = np.linalg.norm(baseline_aligned - gt_aligned, axis=1)

        mean_errors = sliding_windows(errors_pred, WINDOW_SIZE)
        segments = []

        for seg_idx, start in enumerate(pick_top_segments(mean_errors, WINDOW_SIZE, TOP_K), 1):
            frame_start = int(start)
            frame_end = int(min(start + WINDOW_SIZE, length))
            segment_errors = errors_pred[frame_start:frame_end]
            segment_base = errors_base[frame_start:frame_end]
            seg_info = {
                'segment_id': seg_idx,
                'window_size': WINDOW_SIZE,
                'frame_start': frame_start,
                'frame_end': frame_end,
                'mean_error': float(segment_errors.mean()),
                'max_error': float(segment_errors.max()),
                'mean_baseline_error': float(segment_base.mean()),
                'max_baseline_error': float(segment_base.max()),
            }

            plot_path = PLOT_ROOT / f'{name}_seq09_segment{seg_idx:02d}.png'
            marker_frames = FRAME_MARKERS.get(name, {}).get(seg_idx, [])
            plot_segment(name, frame_start, WINDOW_SIZE, gt_aligned, baseline_aligned, pred_aligned, plot_path, marker_frames)
            seg_info['plot_path'] = str(plot_path.relative_to(ROOT))
            seg_info['highlight_frames'] = marker_frames
            segments.append(seg_info)

        results[name] = {
            'trajectory_path': str(path.relative_to(ROOT)),
            'window_size': WINDOW_SIZE,
            'top_segments': segments,
        }

    output_path = OUTPUT_ROOT / 'seq09_drop025_035_segments.json'
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    csv_path = OUTPUT_ROOT / 'seq09_drop_segments.csv'
    with csv_path.open('w', encoding='utf-8') as f:
        headers = ['experiment', 'segment_id', 'frame_start', 'frame_end', 'mean_error', 'max_error', 'mean_baseline_error', 'max_baseline_error', 'plot_path', 'highlight_frames']
        f.write(','.join(headers) + '\n')
        for name, info in results.items():
            for seg in info['top_segments']:
                highlight = ';'.join(map(str, seg.get('highlight_frames', [])))
                row = [
                    name,
                    str(seg['segment_id']),
                    str(seg['frame_start']),
                    str(seg['frame_end']),
                    f"{seg['mean_error']:.4f}",
                    f"{seg['max_error']:.4f}",
                    f"{seg['mean_baseline_error']:.4f}",
                    f"{seg['max_baseline_error']:.4f}",
                    seg['plot_path'],
                    highlight,
                ]
                f.write(','.join(row) + '\n')


if __name__ == '__main__':
    main()
