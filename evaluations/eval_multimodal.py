import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluations.options import BotVIOOptions
from datasets import KITTIOdomPointDataset
from train_multimodal import (
    MultimodalModel,
    build_file_list,
    collate_fn,
    prepare_batch,
    relative_pose,
)


def _parse_sequences(seq_text):
    if not seq_text:
        return None
    sequences = []
    for item in seq_text.split(","):
        token = item.strip()
        if not token:
            continue
        if token.isdigit():
            token = f"{int(token):02d}"
        sequences.append(token)
    return sequences


def _evaluate(model, loader, device, opts):
    pose_cache = {}
    totals = {
        "rot_mae": 0.0,
        "rot_rmse": 0.0,
        "rot_angle_deg": 0.0,
        "trans_mae": 0.0,
        "trans_rmse": 0.0,
        "trans_l2": 0.0,
    }
    count = 0
    try:
        total_samples = len(loader.dataset)
    except (TypeError, AttributeError):
        total_samples = None
    progress_interval = max(1, total_samples // 100) if total_samples else 50

    with torch.no_grad():
        for batch in loader:
            inputs = prepare_batch(batch, device, opts)
            rot_pred, trans_pred = model(inputs)
            rot_vec = rot_pred.squeeze(0).detach().cpu().numpy()
            trans_vec = trans_pred.squeeze(0).detach().cpu().numpy()

            seq, frame = batch["filename"].split()[:2]
            rot_gt, trans_gt = relative_pose(pose_cache, opts.data_path, seq, int(frame))

            rot_diff = rot_vec - rot_gt
            trans_diff = trans_vec - trans_gt

            totals["rot_mae"] += float(np.mean(np.abs(rot_diff)))
            totals["rot_rmse"] += float(np.mean(rot_diff ** 2))
            totals["rot_angle_deg"] += float(np.linalg.norm(rot_diff) * (180.0 / np.pi))
            totals["trans_mae"] += float(np.mean(np.abs(trans_diff)))
            totals["trans_rmse"] += float(np.mean(trans_diff ** 2))
            totals["trans_l2"] += float(np.linalg.norm(trans_diff))
            count += 1

            should_print = count == 1 or (count % progress_interval == 0)
            if total_samples:
                if count == total_samples:
                    should_print = True
                if should_print:
                    percent = (count / total_samples) * 100.0
                    print(
                        f"Progress: {count}/{total_samples} ({percent:.1f}%)",
                        flush=True,
                    )
            elif should_print:
                print(f"Processed {count} samples", flush=True)

    if count == 0:
        raise RuntimeError("No samples evaluated")
    totals["rot_mae"] /= count
    totals["rot_rmse"] = float(np.sqrt(totals["rot_rmse"] / count))
    totals["rot_angle_deg"] /= count
    totals["trans_mae"] /= count
    totals["trans_rmse"] = float(np.sqrt(totals["trans_rmse"] / count))
    totals["trans_l2"] /= count
    return totals, count


def main():
    opts = BotVIOOptions().parse()
    sequences = _parse_sequences(opts.eval_sequences)

    file_list = build_file_list(opts.data_path, sequences=sequences)
    dataset = KITTIOdomPointDataset(
        opts.data_path,
        file_list,
        opts.height,
        opts.width,
        frame_idxs=[0],
        num_scales=1,
        is_train=False,
        img_ext=".png",
        pointcloud_path=opts.pointcloud_path or os.path.join(opts.data_path, "pointclouds"),
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

    device = torch.device("cuda" if torch.cuda.is_available() and not opts.no_cuda else "cpu")
    model = MultimodalModel(opts).to(device)

    checkpoint = torch.load(opts.checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    metrics, count = _evaluate(model, loader, device, opts)

    seq_display = sequences if sequences else "all"
    print(f"Evaluated sequences: {seq_display}")
    print(f"Samples evaluated: {count}")
    print(f"Rotation MAE: {metrics['rot_mae']:.6f} rad")
    print(f"Rotation RMSE: {metrics['rot_rmse']:.6f} rad")
    print(f"Rotation angle error: {metrics['rot_angle_deg']:.6f} deg")
    print(f"Translation MAE: {metrics['trans_mae']:.6f} m")
    print(f"Translation RMSE: {metrics['trans_rmse']:.6f} m")
    print(f"Translation L2: {metrics['trans_l2']:.6f} m")

    if opts.metrics_output:
        output = {"samples": count, "metrics": metrics, "sequences": sequences}
        with open(opts.metrics_output, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        print(f"Metrics written to {opts.metrics_output}")


if __name__ == "__main__":
    main()
