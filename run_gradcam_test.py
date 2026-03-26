import os
import sys

# Ensure output directory exists
os.makedirs('results/visualizations', exist_ok=True)

# Add modules to path
sys.path.insert(0, os.path.dirname(__file__))

# Import and run
from modules.detection.gradcam import YOLOGradCAM

print("Loading YOLO model...")
cam = YOLOGradCAM('models/yolo/best.pt')

print("Processing test image...")
test_img = 'data/processed/merged/test/images/P010001.jpg'

if not os.path.exists(test_img):
    print(f"ERROR: Test image not found: {test_img}")
    sys.exit(1)

heatmap, raw = cam.generate_heatmap(test_img)

import cv2
output_path = 'results/visualizations/gradcam_test.jpg'
cv2.imwrite(output_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

print(f'✓ Grad-CAM test saved to {output_path}')
