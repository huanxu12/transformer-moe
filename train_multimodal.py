import argparse
import math
import os
import pickle
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from evaluations.options import BotVIOOptions
from datasets import KITTIOdomPointDataset
from networks import VisualEncoder, IMUEncoder, PointEncoder, Trans_Fusion, PoseRegressor



def _parse_sequences(seq_text):
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


def collate_fn(batch):
    return batch[0]

def build_file_list(data_root, sequences=None):
    seq_root = Path(data_root) / "sequences"
    pose_root = Path(data_root) / "poses"
    file_list = []
    for seq_dir in sorted(seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq = seq_dir.name
        if sequences and seq not in sequences:
            continue
        pose_path = pose_root / f"{seq}.txt"
        if not pose_path.exists():
            continue
        num_frames = sum(1 for _ in open(pose_path, 'r'))
        left_dir = seq_dir / "image_2"
        for img_path in sorted(left_dir.glob('*.png')):
            frame = int(img_path.stem)
            if frame + 1 >= num_frames:
                continue
            file_list.append(f"{int(seq):02d} {frame:06d} l")
    if not file_list:
        raise RuntimeError("No valid training frames found")
    return file_list


def load_poses(pose_path):
    poses = []
    with open(pose_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            mat = np.eye(4)
            mat[:3, :4] = np.reshape(vals, (3, 4))
            poses.append(mat)
    return poses


def relative_pose(cache, data_root, seq, idx):
    if seq not in cache:
        pose_path = Path(data_root) / "poses" / f"{seq}.txt"
        cache[seq] = load_poses(pose_path)
    mats = cache[seq]
    T0 = mats[idx]
    T1 = mats[idx + 1]
    rel = np.linalg.inv(T0) @ T1
    rot = rel[:3, :3]
    trans = rel[:3, 3]
    rotvec = rotation_matrix_to_axis_angle(rot)
    return rotvec, trans


def rotation_matrix_to_axis_angle(R):
    trace = np.trace(R)
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ], dtype=np.float32) / (2 * math.sin(theta))
    return axis * theta


def to_device(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    return torch.as_tensor(tensor, dtype=torch.float32, device=device)


class MultimodalModel(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.visual_encoder = VisualEncoder(out_dim=opts.v_f_len)
        self.point_encoder = PointEncoder(in_channels=4, out_dim=opts.v_f_len)
        self.imu_encoder = IMUEncoder(out_dim=opts.i_f_len)
        self.fusion = Trans_Fusion(dim=opts.v_f_len, imu_dim=opts.i_f_len, point_dim=opts.v_f_len)
        self.pose_head = PoseRegressor(in_dim=opts.v_f_len, hidden_dim=256)

    def forward(self, inputs):
        visual = inputs['visual']
        point = inputs['point']
        point_valid = inputs['point_valid']
        imu = inputs['imu']

        visual_feat = self.visual_encoder(visual)
        if point_valid.any():
            point_feat = self.point_encoder(point)
        else:
            point_feat = torch.zeros(visual_feat.shape, device=visual_feat.device)
        imu_feat = self.imu_encoder(imu)
        fused = self.fusion(visual_feat, imu_feat, point_feat, point_valid.float())
        rot, trans = self.pose_head(fused)
        return rot, trans


def prepare_batch(inputs, device, opts):
    color = inputs[("color", 0, 0)]
    if not isinstance(color, torch.Tensor):
        color = torch.as_tensor(np.array(color), dtype=torch.float32)
        color = color.permute(2, 0, 1) / 255.0
    if color.dim() == 3:
        visual = color.unsqueeze(0)
    elif color.dim() == 4:
        visual = color
    else:
        raise ValueError(f"Unexpected color tensor shape: {color.shape}")
    visual = visual.to(device, dtype=torch.float32)

    imu_data = inputs['imu']
    imu = torch.as_tensor(imu_data, dtype=torch.float32)
    if imu.dim() == 2:
        imu = imu.unsqueeze(0)
    imu = imu.to(device)

    pc_tensor = inputs[("pointcloud", 0, 0)]
    if not isinstance(pc_tensor, torch.Tensor):
        pc_tensor = torch.as_tensor(pc_tensor, dtype=torch.float32)
    if pc_tensor.dim() == 2:
        pc_tensor = pc_tensor.unsqueeze(0)
    pc_tensor = pc_tensor.to(device)
    channels = pc_tensor.shape[-1] if pc_tensor.dim() >= 3 else 4

    pc_valid_raw = inputs['pc_valid']
    if isinstance(pc_valid_raw, torch.Tensor):
        pc_valid_flag = bool(pc_valid_raw.item())
    else:
        pc_valid_flag = bool(pc_valid_raw)

    pc_count_raw = inputs['pc_count']
    if isinstance(pc_count_raw, torch.Tensor):
        pc_count = int(pc_count_raw.item())
    else:
        pc_count = int(pc_count_raw)

    if pc_valid_flag and pc_count > 0:
        point = pc_tensor[:, :pc_count, :]
        point_valid = torch.ones((1, 1), device=device)
    else:
        point = torch.zeros((1, opts.pc_max_points, channels), device=device)
        point_valid = torch.zeros((1, 1), device=device)

    return {
        'visual': visual,
        'imu': imu,
        'point': point,
        'point_valid': point_valid
    }

def train_epoch(model, loader, optimizer, scheduler, device, pose_cache, opts):
    model.train()
    total_loss = 0.0
    criterion = nn.L1Loss()
    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        optimizer.zero_grad()
        inputs = prepare_batch(batch, device, opts)
        rot_pred, trans_pred = model(inputs)

        seq, frame = batch['filename'].split()[:2]
        rot_gt, trans_gt = relative_pose(pose_cache, opts.data_path, seq, int(frame))
        rot_gt = torch.from_numpy(rot_gt).to(device).unsqueeze(0)
        trans_gt = torch.from_numpy(trans_gt).to(device).unsqueeze(0)

        loss_rot = criterion(rot_pred, rot_gt)
        loss_trans = criterion(trans_pred, trans_gt)
        loss = loss_trans + 0.1 * loss_rot
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    progress.close()
    return total_loss / max(1, len(loader))


def main():
    opts = BotVIOOptions().parse()
    device = torch.device('cuda' if torch.cuda.is_available() and not opts.no_cuda else 'cpu')

    train_sequences = _parse_sequences(opts.train_sequences)
    file_list = build_file_list(opts.data_path, sequences=train_sequences)

    dataset = KITTIOdomPointDataset(
        opts.data_path,
        file_list,
        opts.height,
        opts.width,
        frame_idxs=[0],
        num_scales=1,
        is_train=True,
        img_ext='.png',
        pointcloud_path=opts.pointcloud_path or os.path.join(opts.data_path, 'pointclouds'),
        pc_max_points=opts.pc_max_points,
        pc_min_range=opts.pc_min_range,
        pc_max_range=opts.pc_max_range,
        return_intensity=True
    )

    if opts.train_batch_size != 1:
        print("[train_multimodal] train_batch_size>1 is not supported with current collate_fn; falling back to 1.")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=opts.num_workers,
        collate_fn=collate_fn
    )

    model = MultimodalModel(opts).to(device)

    checkpoint_state = None
    if opts.finetune_checkpoint:
        ckpt_path = Path(opts.finetune_checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint_state = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint_state.get('model', checkpoint_state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {ckpt_path}")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    if opts.freeze_visual:
        model.visual_encoder.requires_grad_(False)
        print("[train_multimodal] Visual encoder frozen")
    if opts.freeze_point:
        model.point_encoder.requires_grad_(False)
        print("[train_multimodal] Point encoder frozen")
    if opts.freeze_imu:
        model.imu_encoder.requires_grad_(False)
        print("[train_multimodal] IMU encoder frozen")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after freezing modules.")

    optimizer = AdamW(trainable_params, lr=opts.learning_rate, weight_decay=opts.weight_decay)
    if opts.resume_optimizer and checkpoint_state and 'optimizer' in checkpoint_state:
        optimizer.load_state_dict(checkpoint_state['optimizer'])
        print("[train_multimodal] Optimizer state resumed")
    elif opts.resume_optimizer and not checkpoint_state:
        print("[train_multimodal] --resume_optimizer specified but no checkpoint was loaded")

    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, len(loader)))

    pose_cache = {}
    history = []

    csv_writer = None
    csv_handle = None
    if opts.log_csv:
        log_path = Path(opts.log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        csv_handle = log_path.open('w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(['epoch', 'loss'])

    output_path = Path(opts.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def dump_checkpoint(target_path):
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history
        }, target_path)

    for epoch in range(opts.num_epochs):
        loss = train_epoch(model, loader, optimizer, scheduler, device, pose_cache, opts)
        history.append({'epoch': epoch + 1, 'loss': loss})
        print(f"Epoch {epoch + 1}/{opts.num_epochs}: loss={loss:.4f}")
        if csv_writer:
            csv_writer.writerow([epoch + 1, f"{loss:.6f}"])
            csv_handle.flush()
        if opts.save_every_epoch > 0 and (epoch + 1) % opts.save_every_epoch == 0:
            interim_name = output_path.with_name(f"{output_path.stem}_epoch{epoch + 1}{output_path.suffix}")
            dump_checkpoint(interim_name)
            print(f"  Saved interim checkpoint to {interim_name}")

    if csv_handle:
        csv_handle.close()

    dump_checkpoint(output_path)
    history_path = output_path.with_suffix(output_path.suffix + '.history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training completed. Checkpoint saved to {output_path}")
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    main()
