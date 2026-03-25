"""
CargoGuard SENTINEL - Kaggle Training Script
Copy this to a Kaggle notebook with GPU enabled.

Instructions:
1. Go to https://www.kaggle.com/notebooks
2. Create a new notebook
3. Enable GPU: Settings > Accelerator > GPU P100/T4
4. Upload your dataset or use the SIXray dataset from Kaggle
5. Copy-paste this code and run
"""

# ============================================================
# CELL 1: Install packages
# ============================================================
# !pip install ultralytics -q
# !pip install grad-cam -q

# ============================================================
# CELL 2: Imports and setup
# ============================================================
from ultralytics import YOLO
import torch
import os

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ============================================================
# CELL 3: Dataset setup
# ============================================================
# Option A: Using Kaggle SIXray dataset
# Dataset: https://www.kaggle.com/datasets/khanhbtq99/sixray

DATASET_PATH = '/kaggle/input/sixray/sixray_v3'

# Create data.yaml for training
yaml_content = f'''
path: {DATASET_PATH}
train: train/images
val: valid/images
test: test/images

nc: 5
names: ['Gun', 'Knife', 'Pliers', 'Scissors', 'Wrench']
'''

with open('/kaggle/working/data.yaml', 'w') as f:
    f.write(yaml_content)

print("data.yaml created")

# ============================================================
# CELL 4: TRAIN YOLOv8 (MAIN TRAINING CELL)
# ============================================================
# Load pretrained YOLOv8-small
model = YOLO('yolov8s.pt')

# Fine-tune on X-ray security dataset
results = model.train(
    data='/kaggle/working/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,              # GPU 0
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
    project='/kaggle/working/runs',
    name='cargoguard_v1',
    exist_ok=True,
    pretrained=True,
    verbose=True
)

print(f'Best mAP50: {results.results_dict["metrics/mAP50(B)"]:.3f}')
print(f'Model saved to: /kaggle/working/runs/cargoguard_v1/weights/best.pt')

# ============================================================
# CELL 5: Monitor training (optional)
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt

results_csv = '/kaggle/working/runs/cargoguard_v1/results.csv'
if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0,0].plot(df['epoch'], df['metrics/mAP50(B)'], 'b-')
    axes[0,0].set_title('mAP@0.5')
    axes[0,0].set_xlabel('Epoch')
    
    axes[0,1].plot(df['epoch'], df['metrics/precision(B)'], 'g-')
    axes[0,1].set_title('Precision')
    axes[0,1].set_xlabel('Epoch')
    
    axes[1,0].plot(df['epoch'], df['metrics/recall(B)'], 'r-')
    axes[1,0].set_title('Recall')
    axes[1,0].set_xlabel('Epoch')
    
    axes[1,1].plot(df['epoch'], df['train/box_loss'], 'k-', label='train')
    axes[1,1].plot(df['epoch'], df['val/box_loss'], 'b--', label='val')
    axes[1,1].set_title('Box Loss')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_curves.png', dpi=150)
    plt.show()

# ============================================================
# CELL 6: Evaluate on test set
# ============================================================
model = YOLO('/kaggle/working/runs/cargoguard_v1/weights/best.pt')

metrics = model.val(
    data='/kaggle/working/data.yaml',
    split='test',
    imgsz=640,
    conf=0.25,
    iou=0.45,
    save_json=True,
    plots=True
)

print('=' * 50)
print('TEST SET RESULTS')
print('=' * 50)
print(f'mAP@0.5:      {metrics.box.map50:.4f}')
print(f'mAP@0.5:0.95: {metrics.box.map:.4f}')
print(f'Precision:    {metrics.box.mp:.4f}')
print(f'Recall:       {metrics.box.mr:.4f}')
print()
print('Per-class mAP:')
class_names = ['Gun', 'Knife', 'Pliers', 'Scissors', 'Wrench']
for name, ap in zip(class_names, metrics.box.ap50):
    print(f'  {name:12s}: {ap:.4f}')

# ============================================================
# CELL 7: Export to ONNX (optional)
# ============================================================
# model.export(format='onnx', imgsz=640, opset=12, simplify=True)
# print('ONNX export complete')

# ============================================================
# Download best.pt from Kaggle Output tab after training!
# ============================================================
