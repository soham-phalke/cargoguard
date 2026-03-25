"""
Dataset Statistics - CargoGuard SENTINEL
Verify dataset class distribution before training.
"""

from pathlib import Path
from collections import Counter


def dataset_stats(data_root: str, class_names: list = None):
    """Analyze dataset statistics for YOLO format data."""
    data_root = Path(data_root)

    if class_names is None:
        class_names = ['Gun', 'Knife', 'Pliers', 'Scissors', 'Wrench']

    print("=" * 60)
    print("DATASET STATISTICS - CargoGuard SENTINEL")
    print("=" * 60)
    print(f"Dataset path: {data_root}")
    print()

    total_images = 0
    total_objects = 0

    for split in ['train', 'valid', 'val', 'test', 'test_easy', 'test_hard', 'test_hidden']:
        lbl_dir = data_root / split / 'labels'
        if not lbl_dir.exists():
            continue

        counts = Counter()
        num_images = 0
        num_objects = 0
        empty_labels = 0

        for lbl_file in lbl_dir.glob('*.txt'):
            num_images += 1
            content = lbl_file.read_text().strip()

            if content:
                for line in content.splitlines():
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        counts[cls] += 1
                        num_objects += 1
            else:
                empty_labels += 1

        total_images += num_images
        total_objects += num_objects

        print(f"--- {split.upper()} ---")
        print(f"  Images: {num_images}")
        print(f"  Objects: {num_objects}")
        print(f"  Empty labels: {empty_labels}")
        print()
        print("  Class distribution:")
        for cls_id in sorted(counts.keys()):
            name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
            count = counts[cls_id]
            pct = (count / num_objects * 100) if num_objects > 0 else 0
            bar = '#' * int(pct / 2)
            print(f"    {cls_id:2d} {name:12s}: {count:5d} ({pct:5.1f}%) |{bar}")
        print()

    print("=" * 60)
    print(f"TOTAL IMAGES:  {total_images}")
    print(f"TOTAL OBJECTS: {total_objects}")
    print("=" * 60)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        default_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'sixray_v3'
        data_path = str(default_path)

    # Auto-detect class names from data.yaml if available
    yaml_path = Path(data_path) / 'data.yaml'
    class_names = None

    if yaml_path.exists():
        import yaml
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    if isinstance(data['names'], dict):
                        class_names = [data['names'][i] for i in sorted(data['names'].keys())]
                    else:
                        class_names = data['names']
        except:
            pass

    # Default to PIDray 12 classes if not found
    if class_names is None:
        class_names = ['Baton', 'Pliers', 'Hammer', 'Powerbank', 'Scissors', 'Wrench',
                      'Gun', 'Bullet', 'Sprayer', 'HandCuffs', 'Knife', 'Lighter']

    dataset_stats(data_path, class_names)
