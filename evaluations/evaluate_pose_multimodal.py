import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from evaluations.options import BotVIOOptions
from datasets import KITTIOdomPointDataset
from train_multimodal import (
    MultimodalModel,
    build_file_list,
    collate_fn,
    prepare_batch,
)


PoseSequence = Dict[str, Dict[int, np.ndarray]]


def _parse_sequences(seq_text: str):
    if not seq_text:
        return None
    sequences = []
    for item in seq_text.split(','):
        token = item.strip()
        if not token:
            continue
        if token.isdigit():
            token = f"{int(token):02d}"
        sequences.append(token)
    return sequences


def _axis_angle_to_matrix(vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = vector / theta
    x, y, z = axis
    skew = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    eye = np.eye(3, dtype=np.float64)
    return eye + sin_theta * skew + (1.0 - cos_theta) * (skew @ skew)


def _compose_transform(rot_vec: np.ndarray, trans_vec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _axis_angle_to_matrix(rot_vec)
    transform[:3, 3] = trans_vec.astype(np.float64)
    return transform


def _evaluate_trajectories(model: MultimodalModel,
                           loader: DataLoader,
                           device: torch.device,
                           opts) -> PoseSequence:
    seq_states: Dict[str, Dict[str, object]] = {}
    dataset_size = len(loader.dataset)
    progress_interval = max(1, dataset_size // 100) if dataset_size else 50
    processed = 0
    with torch.no_grad():
        for batch in loader:
            processed += 1
            inputs = prepare_batch(batch, device, opts)
            rot_pred, trans_pred = model(inputs)
            rot_vec = rot_pred.squeeze(0).detach().cpu().numpy()
            trans_vec = trans_pred.squeeze(0).detach().cpu().numpy()

            seq, frame_str = batch['filename'].split()[:2]
            frame_idx = int(frame_str)

            state = seq_states.get(seq)
            if state is None:
                state = {
                    'current_pose': np.eye(4, dtype=np.float64),
                    'poses': {frame_idx: np.eye(4, dtype=np.float64)},
                    'last_frame': frame_idx,
                }
                seq_states[seq] = state

            expected = state['last_frame']
            if frame_idx != expected:
                raise RuntimeError(
                    f"Non-consecutive frame index for sequence {seq}: "
                    f"expected {expected}, got {frame_idx}"
                )

            step_transform = _compose_transform(rot_vec, trans_vec)
            state['current_pose'] = state['current_pose'] @ step_transform
            next_frame = frame_idx + 1
            state['poses'][next_frame] = state['current_pose'].copy()
            state['last_frame'] = next_frame

            should_print = processed == 1 or processed == dataset_size or (processed % progress_interval == 0)
            if should_print:
                percent = (processed / dataset_size) * 100.0 if dataset_size else 0.0
                print(
                    f"Processed {processed}/{dataset_size} samples ({percent:.1f}%)",
                    flush=True,
                )

    trajectories: PoseSequence = {}
    for seq, state in seq_states.items():
        poses_map: Dict[int, np.ndarray] = state['poses']  # type: ignore[assignment]
        trajectories[seq] = poses_map
    return trajectories


def _write_results(trajectories: PoseSequence, results_dir: Path, overwrite: bool = False) -> List[str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[str] = []
    for seq in sorted(trajectories.keys()):
        poses_map = trajectories[seq]
        indices = sorted(poses_map.keys())
        if not indices:
            continue
        # Ensure frame indices are consecutive for KITTI format
        start = indices[0]
        expected = start
        for idx in indices:
            if idx != expected:
                raise RuntimeError(
                    f"Sequence {seq} has missing pose for frame {expected} (found {idx})"
                )
            expected += 1
        output_path = results_dir / f"{seq}.txt"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing file {output_path}. "
                "Pass --overwrite_results to replace it."
            )
        with open(output_path, 'w', encoding='utf-8') as handle:
            for idx in indices:
                pose = poses_map[idx]
                mat = pose[:3, :4].reshape(-1)
                line = ' '.join(f"{val:.8f}" for val in mat)
                handle.write(line + '\n')
        summaries.append(f"seq {seq}: {len(indices)} poses -> {output_path}")
    return summaries


def main():
    options = BotVIOOptions()
    options.parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory to store KITTI-format trajectory files.')
    options.parser.add_argument(
        '--overwrite_results',
        action='store_true',
        help='Allow overwriting existing trajectory files in the results directory.')
    opts = options.parse()

    sequences = _parse_sequences(opts.eval_sequences)
    file_list = build_file_list(opts.data_path, sequences=sequences)
    if not file_list:
        raise RuntimeError('No evaluation files found. Check --data_path and --eval_sequences.')

    print(f"Evaluating sequences: {sequences if sequences else 'all'} (files={len(file_list)})")
    dataset = KITTIOdomPointDataset(
        opts.data_path,
        file_list,
        opts.height,
        opts.width,
        frame_idxs=[0],
        num_scales=1,
        is_train=False,
        img_ext='.png',
        pointcloud_path=opts.pointcloud_path or os.path.join(opts.data_path, 'pointclouds'),
        pc_max_points=opts.pc_max_points,
        pc_min_range=opts.pc_min_range,
        pc_max_range=opts.pc_max_range,
        return_intensity=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opts.num_workers,
        collate_fn=collate_fn,
    )

    device = torch.device('cuda' if torch.cuda.is_available() and not opts.no_cuda else 'cpu')
    model = MultimodalModel(opts).to(device)

    checkpoint = torch.load(opts.checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    trajectories = _evaluate_trajectories(model, loader, device, opts)
    summaries = _write_results(trajectories, Path(opts.results_dir), overwrite=opts.overwrite_results)

    print('Trajectory files written:')
    for summary in summaries:
        print(f'  {summary}')


if __name__ == '__main__':
    main()
