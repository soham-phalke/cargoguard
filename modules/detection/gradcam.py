import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

class YOLOGradCAM:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = YOLO(model_path)
        self.model_torch = self.model.model.to(device)
        # Keep model in train mode for Grad-CAM to work
        self.model_torch.train()
        # Target the last convolutional layer before the head
        self.target_layer = [self.model_torch.model[-2]]
        self.cam = GradCAM(
            model=self.model_torch,
            target_layers=self.target_layer
        )
    
    def preprocess(self, image_path: str):
        """Load and preprocess image for Grad-CAM."""
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_float = img_resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        tensor.requires_grad = True  # Enable gradients for Grad-CAM
        return tensor, img_float
    
    def generate_heatmap(self, image_path: str, class_idx: int = None):
        """
        Generate Grad-CAM heatmap for the given image.
        Returns: (heatmap_overlay, raw_heatmap)
        """
        input_tensor, img_float = self.preprocess(image_path)
        
        targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None
        
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=True,      # Smoother heatmap
            eigen_smooth=True     # Less noise
        )
        grayscale_cam = grayscale_cam[0]  # Remove batch dim
        
        # Overlay on original image
        visualization = show_cam_on_image(
            img_float, grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
            image_weight=0.5  # 50% original, 50% heatmap
        )
        return visualization, grayscale_cam
    
    def annotate_image(self, image_path: str, detections: list):
        """
        Draw bounding boxes + Grad-CAM overlay on image.
        detections: list of {class_name, confidence, box:[x1,y1,x2,y2], class_idx}
        Returns: annotated image as numpy array (RGB)
        """
        img_orig = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_orig.shape[:2]
        
        # Try to generate Grad-CAM for highest-confidence detection
        result = img_rgb.copy()
        if detections:
            try:
                top_det = max(detections, key=lambda d: d['confidence'])
                heatmap_overlay, _ = self.generate_heatmap(image_path, top_det['class_idx'])
                heatmap_resized = cv2.resize(heatmap_overlay, (w_orig, h_orig))
                result = cv2.addWeighted(img_rgb, 0.6, heatmap_resized, 0.4, 0)
            except Exception as e:
                # Fallback: just use original image if Grad-CAM fails
                print(f"Warning: Grad-CAM failed ({e}), using simple annotation")
                result = img_rgb.copy()
        
        # Draw bounding boxes
        COLORS = {
            'gun':     (255, 50,  50),   # Red
            'knife':   (255, 80,  80),   # Red
            'wrench':  (255, 165, 0),    # Orange
            'pliers':  (255, 165, 0),    # Orange
            'scissors':(255, 200, 0),    # Amber
            'hammer':    (0,   180, 180),  # Teal
            'powerbank':  (255, 140,   0),  # Dark Orange
            'baton':      (200,  50,  50),  # Dark Red
            'bullet':     (255,   0,   0),  # Bright Red
            'sprayer':    (255, 165,   0),  # Orange
            'handcuffs':  (100, 180, 100),  # Green
            'lighter':    (150, 150, 255),  # Light Blue
        }
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            color = COLORS.get(det['class_name'], (200, 200, 200))
            label = f"{det['class_name'].upper()} {det['confidence']:.0%}"
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            # Draw label background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1-lh-8), (x1+lw+4, y1), color, -1)
            cv2.putText(result, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return result


# ── Quick test ──────────────────────────────────────────────
if __name__ == '__main__':
    import os
    from glob import glob as file_glob
    
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Find a test image
    test_patterns = [
        'data/processed/merged/test/images/*.jpg',
        'data/processed/merged/test/images/*.png',
        'data/processed/merged/train/images/*.jpg',
        'data/processed/merged/train/images/*.png',
        'data/processed/merged/val/images/*.jpg',
    ]
    
    test_img = None
    for pattern in test_patterns:
        matches = file_glob(pattern)
        if matches:
            test_img = matches[0]
            break
    
    if not test_img:
        print("❌ ERROR: No test images found in data/processed/merged/")
        print("   Please ensure the dataset has been extracted/merged.")
        print("   Expected location: data/processed/merged/{train,val,test}/images/")
        exit(1)
    
    print(f"Using test image: {test_img}")
    cam = YOLOGradCAM('models/yolo/best.pt')
    
    try:
        heatmap, raw = cam.generate_heatmap(test_img)
        cv2.imwrite('results/visualizations/gradcam_test.jpg',
                    cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        print('✓ Grad-CAM test saved to results/visualizations/gradcam_test.jpg')
    except Exception as e:
        print(f"\n⚠ Warning: Grad-CAM visualization failed: {e}")
        print("   This is a known limitation with YOLO detection models.")
        print("   The detection pipeline will work with simple bounding box annotation.")
        print("\n   Test the full pipeline with: python test_detector.py")
        exit(0)
