"""
PIDray to YOLO Converter - CargoGuard SENTINEL
Convert PIDray COCO JSON annotations to YOLO format.

PIDray Dataset: 12 classes, 29,457 images
Format: COCO JSON → YOLO txt files
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image


# PIDray class mapping (COCO category_id → YOLO class_id)
PIDRAY_CLASS_MAP = {
    1:  0,   # Baton
    2:  1,   # Pliers
    3:  2,   # Hammer
    4:  3,   # Powerbank
    5:  4,   # Scissors
    6:  5,   # Wrench
    7:  6,   # Gun
    8:  7,   # Bullet
    9:  8,   # Sprayer
    10: 9,   # HandCuffs
    11: 10,  # Knife
    12: 11,  # Lighter
}

CLASS_NAMES = ['Baton', 'Pliers', 'Hammer', 'Powerbank', 'Scissors', 'Wrench',
               'Gun', 'Bullet', 'Sprayer', 'HandCuffs', 'Knife', 'Lighter']


def convert_bbox_coco_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [cx, cy, w, h] (normalized).

    Args:
        bbox: [x, y, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        [cx, cy, w, h] normalized to 0-1
    """
    x, y, w, h = bbox

    # Convert to center coordinates
    cx = (x + w / 2) / img_width
    cy = (y + h / 2) / img_height
    w = w / img_width
    h = h / img_height

    # Clip to valid range
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return cx, cy, w, h


def convert_pidray_to_yolo(pidray_root: str, output_root: str):
    """
    Convert PIDray COCO JSON annotations to YOLO format.

    Args:
        pidray_root: Path to PIDray dataset root
        output_root: Path to output YOLO dataset
    """
    pidray_root = Path(pidray_root)
    output_root = Path(output_root)

    # Create output directories
    splits = {
        'train': 'xray_train.json',
        'test_easy': 'xray_test_easy.json',
        'test_hard': 'xray_test_hard.json',
        'test_hidden': 'xray_test_hidden.json'
    }

    for split in splits.keys():
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Process each split
    total_images = 0
    total_objects = 0

    for split_name, json_file in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing {split_name}...")
        print('='*60)

        json_path = pidray_root / 'annotations' / json_file

        if not json_path.exists():
            print(f"  ⚠ Skipping: {json_path} not found")
            continue

        # Load COCO JSON
        with open(json_path) as f:
            coco_data = json.load(f)

        # Build image_id to filename mapping
        images_dict = {img['id']: img for img in coco_data['images']}

        # Group annotations by image_id
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        # Process each image
        num_images = 0
        num_objects = 0

        for img_id, img_info in tqdm(images_dict.items(), desc=f"  {split_name}"):
            filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']

            # Determine source directory
            if split_name == 'train':
                src_img_path = pidray_root / 'train' / filename
            else:
                # test splits are in easy/hard/hidden folders
                test_type = split_name.replace('test_', '')
                src_img_path = pidray_root / test_type / filename

            if not src_img_path.exists():
                continue

            # Copy image
            dst_img_path = output_root / split_name / 'images' / filename
            shutil.copy(src_img_path, dst_img_path)

            # Convert annotations to YOLO format
            yolo_labels = []
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    category_id = ann['category_id']
                    bbox = ann['bbox']

                    # Map to YOLO class
                    if category_id not in PIDRAY_CLASS_MAP:
                        continue

                    yolo_class = PIDRAY_CLASS_MAP[category_id]

                    # Convert bbox
                    cx, cy, w, h = convert_bbox_coco_to_yolo(bbox, img_width, img_height)

                    yolo_labels.append(f"{yolo_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    num_objects += 1

            # Write YOLO label file
            label_name = Path(filename).stem + '.txt'
            label_path = output_root / split_name / 'labels' / label_name

            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

            num_images += 1

        print(f"  ✓ {split_name}: {num_images} images, {num_objects} objects")
        total_images += num_images
        total_objects += num_objects

    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print('='*60)
    print(f"Total images:  {total_images}")
    print(f"Total objects: {total_objects}")
    print(f"Output: {output_root}")


def create_pidray_data_yaml(output_root: str):
    """Create data.yaml for PIDray dataset."""
    output_root = Path(output_root)

    yaml_content = f"""# PIDray Dataset - YOLO Format
# 12 prohibited item classes

path: {output_root.absolute().as_posix()}
train: train/images
val: test_easy/images
test: test_hard/images

nc: 12
names:
  0: Baton
  1: Pliers
  2: Hammer
  3: Powerbank
  4: Scissors
  5: Wrench
  6: Gun
  7: Bullet
  8: Sprayer
  9: HandCuffs
  10: Knife
  11: Lighter

# Dataset Statistics:
# Train: ~23,732 images
# Test Easy: ~2,831 images
# Test Hard: ~1,466 images
# Test Hidden: ~1,428 images
# Total: ~29,457 images
"""

    yaml_path = output_root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ data.yaml created at {yaml_path}")


if __name__ == '__main__':
    import sys

    # Default paths
    pidray_root = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'pidray'
    output_root = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'pidray_yolo'

    if not pidray_root.exists():
        print(f"ERROR: PIDray dataset not found at {pidray_root}")
        print("Please download PIDray dataset first.")
        sys.exit(1)

    print("=" * 60)
    print("PIDRAY TO YOLO CONVERTER")
    print("=" * 60)
    print(f"Source: {pidray_root}")
    print(f"Output: {output_root}")
    print()

    # Convert dataset
    convert_pidray_to_yolo(str(pidray_root), str(output_root))

    # Create data.yaml
    create_pidray_data_yaml(str(output_root))

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Verify converted data:")
    print(f"   python modules/detection/dataset_stats.py {output_root}")
    print()
    print("2. Train with PIDray:")
    print(f"   Update data.yaml path to: {output_root / 'data.yaml'}")
    print()
