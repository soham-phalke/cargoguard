"""
Local Training Script - CargoGuard SENTINEL
Train YOLOv8 on PIDray dataset (12 classes, 47K images)

Usage:
  python modules/detection/train_local.py

For Kaggle training, use notebooks/kaggle_training.py instead.
"""

from pathlib import Path
from ultralytics import YOLO
import torch


def train_model(data_yaml: str, epochs: int = 50, batch_size: int = 16,
                img_size: int = 640, device: str = None):
    """
    Train YOLOv8 on X-ray security dataset.

    Args:
        data_yaml: Path to data.yaml configuration
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        img_size: Image size for training
        device: 'cpu', 'cuda', or device ID
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("CARGOGUARD SENTINEL - LOCAL TRAINING")
    print("=" * 60)
    print(f"Data:    {data_yaml}")
    print(f"Device:  {device}")
    print(f"Epochs:  {epochs}")
    print(f"Batch:   {batch_size}")
    print(f"ImgSize: {img_size}")
    print()

    if device == 'cpu':
        print("WARNING: Training on CPU will be very slow!")
        print("Recommended: Use Kaggle/Colab with free GPU")
        print("Or reduce epochs to 10-20 for quick test")
        print()

    # Load pretrained YOLOv8-small
    model = YOLO('yolov8s.pt')

    # Training configuration optimized for X-ray images
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        optimizer='SGD',
        lr0=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        # Augmentation for X-ray images
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        flipud=0.0,            # X-rays should not be flipped vertically
        fliplr=0.5,
        degrees=0.0,           # No rotation for X-rays
        translate=0.1,
        scale=0.5,
        hsv_h=0.0,             # X-rays are grayscale
        hsv_s=0.0,
        hsv_v=0.4,             # Only brightness variation
        # Training config
        patience=15,           # Early stopping
        save=True,
        save_period=10,
        project='runs/train',
        name='cargoguard_pidray',
        exist_ok=True,
        pretrained=True,
        verbose=True
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Model saved to: runs/train/cargoguard_pidray/weights/best.pt")
    print()
    print("Next steps:")
    print("1. Copy best.pt to models/yolo/best.pt")
    print("2. Run evaluation: python modules/detection/evaluate.py")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLOv8 on CargoGuard dataset')
    parser.add_argument('--data', type=str, default=None, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda/0)')
    args = parser.parse_args()

    # Default to PIDray dataset
    if args.data is None:
        base_dir = Path(__file__).parent.parent.parent
        args.data = str(base_dir / 'data' / 'processed' / 'pidray_yolo' / 'data.yaml')

    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device
    )
