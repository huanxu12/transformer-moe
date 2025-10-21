import argparse
import shutil
import zipfile
from pathlib import Path


def parse_sequences(seq_text: str):
    seqs = []
    for token in seq_text.split(','):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            token = f"{int(token):02d}"
        seqs.append(token)
    return seqs


def has_velodyne(dest_root: Path, seq: str) -> bool:
    velodyne_dir = dest_root / seq / 'velodyne'
    return velodyne_dir.is_dir() and any(velodyne_dir.glob('*.bin'))


def copy_from_source(src_root: Path, dest_root: Path, seqs):
    copied = []
    for seq in seqs:
        candidates = [
            src_root / 'sequences' / seq / 'velodyne',
            src_root / 'dataset' / 'sequences' / seq / 'velodyne',
            src_root / seq / 'velodyne'
        ]
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob('*.bin')):
                target = dest_root / seq / 'velodyne'
                target.mkdir(parents=True, exist_ok=True)
                for file_path in candidate.glob('*.bin'):
                    dst = target / file_path.name
                    if dst.exists():
                        continue
                    shutil.copy2(file_path, dst)
                copied.append(seq)
                break
    return copied


def extract_from_zip(zip_path: Path, dest_root: Path, seqs):
    try:
        with zipfile.ZipFile(zip_path, 'r') as archive:
            extracted = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                filename = info.filename.replace('\\', '/')
                for seq in seqs:
                    key = f"sequences/{seq}/velodyne/"
                    if key in filename:
                        rel = filename.split('sequences/', 1)[1]
                        target = dest_root / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists():
                            continue
                        with archive.open(info) as src, target.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
                        if seq not in extracted:
                            extracted.append(seq)
            return extracted
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Invalid ZIP archive: {zip_path}") from exc


def main():
    parser = argparse.ArgumentParser(description="Ensure KITTI velodyne bins exist for specified sequences")
    parser.add_argument('--data-root', type=Path, default=Path('data') / 'sequences',
                        help='Target KITTI data/sequences root')
    parser.add_argument('--sequences', type=str, default='09,10',
                        help='Comma separated sequence ids (default: 09,10)')
    parser.add_argument('--src-root', type=Path, default=None,
                        help='Optional existing KITTI odometry root to copy velodyne files from')
    parser.add_argument('--zip', type=Path, action='append', default=[],
                        help='Optional path(s) to KITTI data_odometry_velodyne zip archives')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report actions without copying or extracting')
    args = parser.parse_args()

    seqs = parse_sequences(args.sequences)
    if not seqs:
        raise SystemExit('No sequences specified')

    data_root = Path(args.data_root)
    report = []
    missing = []

    for seq in seqs:
        if has_velodyne(data_root, seq):
            report.append((seq, 'present'))
        else:
            missing.append(seq)
            report.append((seq, 'missing'))

    actions = []
    if missing and not args.dry_run:
        copied = []
        if args.src_root:
            copied = copy_from_source(args.src_root, data_root, missing)
            if copied:
                actions.append(f"Copied sequences {', '.join(copied)} from {args.src_root}")
                missing = [seq for seq in missing if seq not in copied and not has_velodyne(data_root, seq)]
        for zip_path in args.zip:
            if not missing:
                break
            extracted = extract_from_zip(zip_path, data_root, missing)
            if extracted:
                actions.append(f"Extracted {zip_path} for sequences {', '.join(extracted)}")
                missing = [seq for seq in missing if seq not in extracted and not has_velodyne(data_root, seq)]

    print('Velodyne availability:')
    for seq, status in report:
        updated_status = 'present' if has_velodyne(data_root, seq) else status
        print(f"  seq {seq}: {updated_status}")
    if actions:
        print('\nActions performed:')
        for act in actions:
            print(f"  - {act}")
    if missing:
        print('\nSequences still missing point clouds:', ', '.join(missing))
        print('Provide --src-root or --zip with official KITTI data_odometry_velodyne archive to populate them.')


if __name__ == '__main__':
    main()
