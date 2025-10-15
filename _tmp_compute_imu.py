import argparse
import json
from pathlib import Path

import numpy as np

from datasets import KITTIOdomPointDataset
from train_multimodal import build_file_list


def parse_sequences(text):
    if not text:
        return None
    sequences = []
    for item in text.split(','):
        token = item.strip()
        if not token:
            continue
        if token.isdigit():
            token = f"{int(token):02d}"
        sequences.append(token)
    return sequences


def parse_args():
    parser = argparse.ArgumentParser(description="Compute IMU normalization statistics for BotVIO")
    parser.add_argument('--data_path', type=str, default='data',
                        help='dataset root containing sequences/imus directories')
    parser.add_argument('--output', type=str, default=None,
                        help='destination JSON file; defaults to <data_path>/imu_stats.json')
    parser.add_argument('--train_sequences', type=str,
                        default='00,01,02,03,04,05,06,07,08',
                        help='comma-separated KITTI sequence ids to use for statistics')
    parser.add_argument('--height', type=int, default=192,
                        help='image height used when instantiating the dataset loader')
    parser.add_argument('--width', type=int, default=640,
                        help='image width used when instantiating the dataset loader')
    parser.add_argument('--limit', type=int, default=None,
                        help='optional cap on the number of samples to process (debugging)')
    parser.add_argument('--imu_gravity_axis', type=int, default=None,
                        help='axis index (0:x, 1:y, 2:z) to subtract gravity before statistics')
    parser.add_argument('--imu_gravity_value', type=float, default=9.81,
                        help='gravity magnitude to subtract when imu_gravity_axis is set')
    parser.add_argument('--mean_tolerance', type=float, default=0.1,
                        help='max allowed absolute deviation of normalized mean')
    parser.add_argument('--std_tolerance', type=float, default=0.1,
                        help='max allowed absolute deviation of normalized std from 1.0')
    return parser.parse_args()


def apply_gravity_correction(imu_window, axis, gravity_value):
    channels = imu_window.shape[-1]
    if axis < -channels or axis >= channels:
        raise ValueError(
            f"imu_gravity_axis={axis} is outside valid range for tensor with {channels} channels"
        )
    imu_window = np.array(imu_window, dtype=np.float32, copy=True)
    imu_window[..., axis] = imu_window[..., axis] - gravity_value
    return imu_window


def main():
    args = parse_args()
    sequences = parse_sequences(args.train_sequences)
    file_list = build_file_list(args.data_path, sequences=sequences)
    if args.limit:
        file_list = file_list[:args.limit]
    print(f"Collected {len(file_list)} frames for IMU stats")

    dataset = KITTIOdomPointDataset(
        args.data_path,
        file_list,
        args.height,
        args.width,
        frame_idxs=[0],
        num_scales=1,
        is_train=True,
        img_ext='.png'
    )

    imu_windows = []
    gravity_axis = args.imu_gravity_axis
    for idx, sample in enumerate(dataset):
        imu = np.asarray(sample['imu'], dtype=np.float32)
        if gravity_axis is not None:
            imu = apply_gravity_correction(imu, int(gravity_axis), float(args.imu_gravity_value))
        imu_windows.append(imu)
        if args.limit and len(imu_windows) >= args.limit:
            break
    print(f"Loaded {len(imu_windows)} IMU windows")

    if not imu_windows:
        raise RuntimeError('No IMU samples collected; check dataset path and sequences.')

    imu_all = np.vstack(imu_windows)
    mean = imu_all.mean(axis=0)
    std = imu_all.std(axis=0)
    if np.any(std < 1e-6):
        raise ValueError('Computed IMU std contains near-zero values; cannot normalize reliably.')

    stats = {
        'mean': mean.tolist(),
        'std': std.tolist(),
    }

    normalized = (imu_all - mean) / std
    residual_mean = normalized.mean(axis=0)
    residual_std = normalized.std(axis=0)
    mean_deviation = float(np.max(np.abs(residual_mean)))
    std_deviation = float(np.max(np.abs(residual_std - 1.0)))

    print('Residual normalized mean:', residual_mean.tolist())
    print('Residual normalized std:', residual_std.tolist())

    if mean_deviation > args.mean_tolerance:
        raise AssertionError(
            f'Normalized mean deviates by {mean_deviation:.4f}, exceeding tolerance {args.mean_tolerance}'
        )
    if std_deviation > args.std_tolerance:
        raise AssertionError(
            f'Normalized std deviates by {std_deviation:.4f}, exceeding tolerance {args.std_tolerance}'
        )

    output_path = Path(args.output) if args.output else Path(args.data_path) / 'imu_stats.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(stats, handle, indent=2)

    print(f'Saved IMU stats to {output_path}')
    print('mean:', stats['mean'])
    print('std:', stats['std'])


if __name__ == '__main__':
    main()
