import csv
import json
from pathlib import Path
import shutil
import math

ROOT = Path('.').resolve()
CSV_PATH = ROOT / 'results' / 'analysis' / 'seq09_drop_segments.csv'
SEQ_ROOT = ROOT / 'data' / 'sequences' / '09'
POINTCLOUD_ROOT = ROOT / 'data' / 'pointclouds' / '09'
DEST_ROOT = ROOT / 'results' / 'analysis' / 'seq09_segment_assets'
SAMPLE_COUNT = 6  # number of frames sampled per segment (start/end inclusive)

IMAGE_DIR = SEQ_ROOT / 'image_2'
VELODYNE_CANDIDATES = [SEQ_ROOT / 'velodyne', POINTCLOUD_ROOT]
VELODYNE_DIR = next((cand for cand in VELODYNE_CANDIDATES if cand.exists()), None)
if VELODYNE_DIR is None:
    raise FileNotFoundError('Velodyne directory not found; expected data/sequences/09/velodyne or data/pointclouds/09')
OXTS_DIR = SEQ_ROOT / 'oxts'

DEST_ROOT.mkdir(parents=True, exist_ok=True)

index_records = []
summary = {}

with CSV_PATH.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        experiment = row['experiment']
        segment_id = int(row['segment_id'])
        frame_start = int(row['frame_start'])
        frame_end = int(row['frame_end'])
        key = f"{experiment}_seg{segment_id:02d}"

        dest_segment = DEST_ROOT / key
        dest_img = dest_segment / 'image_2'
        dest_velo = dest_segment / 'velodyne'
        dest_oxts = dest_segment / 'oxts'
        for d in (dest_segment, dest_img, dest_velo, dest_oxts):
            d.mkdir(parents=True, exist_ok=True)

        count = frame_end - frame_start
        if count <= 0:
            continue

        step = max(1, math.floor(count / (SAMPLE_COUNT - 1))) if SAMPLE_COUNT > 1 else 1
        selected_frames = list(range(frame_start, frame_end, step))[:SAMPLE_COUNT-1]
        if selected_frames and selected_frames[-1] != frame_end - 1:
            selected_frames.append(frame_end - 1)
        if not selected_frames:
            selected_frames = [frame_start, frame_end - 1] if frame_end - frame_start > 1 else [frame_start]

        copied_frames = []
        missing = {'image': [], 'velodyne': [], 'oxts': []}
        for frame in selected_frames:
            frame_id = f"{frame:06d}"
            src_img = IMAGE_DIR / f"{frame_id}.png"
            src_velo = VELODYNE_DIR / f"{frame_id}.bin"
            src_oxts = OXTS_DIR / f"{frame_id}.txt"

            if src_img.exists():
                shutil.copy2(src_img, dest_img / src_img.name)
            else:
                missing['image'].append(frame)

            if src_velo.exists():
                shutil.copy2(src_velo, dest_velo / src_velo.name)
            else:
                missing['velodyne'].append(frame)

            if src_oxts.exists():
                shutil.copy2(src_oxts, dest_oxts / src_oxts.name)
            else:
                missing['oxts'].append(frame)

            copied_frames.append(frame)

            index_records.append({
                'segment': key,
                'frame': frame,
                'image_path': str((dest_img / src_img.name).relative_to(ROOT)) if src_img.exists() else '',
                'velodyne_path': str((dest_velo / src_velo.name).relative_to(ROOT)) if src_velo.exists() else '',
                'oxts_path': str((dest_oxts / src_oxts.name).relative_to(ROOT)) if src_oxts.exists() else '',
            })

        summary[key] = {
            'experiment': experiment,
            'segment_id': segment_id,
            'frame_start': frame_start,
            'frame_end': frame_end,
            'sampled_frames': copied_frames,
            'missing_files': missing,
            'plot_reference': row.get('plot_path', ''),
            'mean_error': float(row.get('mean_error', 0.0)),
            'max_error': float(row.get('max_error', 0.0)),
        }

index_csv = DEST_ROOT / 'segment_samples_index.csv'
with index_csv.open('w', encoding='utf-8') as f:
    f.write('segment,frame,image_path,velodyne_path,oxts_path\n')
    for rec in index_records:
        f.write(f"{rec['segment']},{rec['frame']},{rec['image_path']},{rec['velodyne_path']},{rec['oxts_path']}\n")

summary_json = DEST_ROOT / 'segment_samples_summary.json'
with summary_json.open('w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Sampled data written under {DEST_ROOT}")
