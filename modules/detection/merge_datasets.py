"""
Fast merge using file copy with absolute paths (Windows compatible)
"""
import os
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def fast_merge(sources, output_root, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    # Use absolute paths
    base_dir = Path(__file__).parent.parent.parent.resolve()
    output_root = base_dir / output_root

    # Create dirs (don't delete existing - allows resume)
    for split in ['train', 'val', 'test']:
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Collect all samples with absolute paths
    all_samples = []
    for src in sources:
        src = base_dir / src
        for split_folder in ['train', 'test_easy', 'test_hard', 'test_hidden']:
            img_dir = src / split_folder / 'images'
            lbl_dir = src / split_folder / 'labels'
            if not img_dir.exists():
                continue
            print(f"Scanning {src.name}/{split_folder}...")
            for img_file in img_dir.glob('*'):
                lbl_file = lbl_dir / (img_file.stem + '.txt')
                if lbl_file.exists():
                    all_samples.append((img_file.resolve(), lbl_file.resolve()))

    print(f'Total samples: {len(all_samples)}')

    # Split
    train_s, temp_s = train_test_split(all_samples, test_size=val_ratio+test_ratio, random_state=seed)
    val_s, test_s = train_test_split(temp_s, test_size=0.5, random_state=seed)

    splits = {'train': train_s, 'val': val_s, 'test': test_s}

    # Copy files (skip locked files)
    for split_name, samples in splits.items():
        print(f'Copying {split_name}: {len(samples)} files...')
        skipped = 0
        copied = 0
        for i, (img_src, lbl_src) in enumerate(samples):
            img_dst = output_root / split_name / 'images' / img_src.name
            lbl_dst = output_root / split_name / 'labels' / lbl_src.name
            try:
                if not img_dst.exists():
                    shutil.copy2(str(img_src), str(img_dst))
                if not lbl_dst.exists():
                    shutil.copy2(str(lbl_src), str(lbl_dst))
                copied += 1
            except (PermissionError, OSError) as e:
                skipped += 1
                continue
            if (i+1) % 5000 == 0:
                print(f'  {i+1}/{len(samples)} done (skipped: {skipped})')
        print(f'{split_name}: {copied} copied, {skipped} skipped')

    print('\n=== MERGE COMPLETE ===')
    print(f'Train: {len(train_s)}')
    print(f'Val: {len(val_s)}')
    print(f'Test: {len(test_s)}')
    print(f'Total: {len(all_samples)}')

if __name__ == '__main__':
    fast_merge(
        sources=['data/processed/pidray_yolo'],
        output_root='data/processed/merged'
    )
