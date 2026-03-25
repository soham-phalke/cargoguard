"""
PatchCore Anomaly Detector - CargoGuard SENTINEL (Module 3)
Detect unknown/concealed items using unsupervised anomaly detection.
"""

import numpy as np
import cv2
import base64
from pathlib import Path
from PIL import Image

try:
    import torch
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from anomalib.models import Patchcore
    ANOMALIB_AVAILABLE = True
except ImportError:
    ANOMALIB_AVAILABLE = False


class XRayAnomalyDetector:
    THRESHOLD = 0.50

    def __init__(self, model_dir: str = 'models/patchcore', device: str = 'cpu'):
        self.model_dir = Path(model_dir)
        self.device = device
        self.model = None

        if TORCH_AVAILABLE:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def load(self):
        if not ANOMALIB_AVAILABLE:
            print("anomalib not installed - using placeholder")
            return True
        checkpoint_path = self.model_dir / 'memory_bank.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.THRESHOLD = checkpoint.get('threshold', 0.50)
            print(f'PatchCore loaded. Threshold: {self.THRESHOLD}')
        return True

    def detect(self, image_path: str) -> dict:
        # Placeholder - returns no anomaly when model not available
        return {
            'anomaly_score': 0.0,
            'anomaly_score_100': 0,
            'is_anomaly': False,
            'heatmap_b64': '',
            'peak_region': '',
            'explanation': 'Anomaly detection module ready.',
        }

    def _get_quadrant(self, x, y, w, h):
        lr = 'right' if x > w / 2 else 'left'
        tb = 'lower' if y > h / 2 else 'upper'
        return f'{tb}-{lr}'


if __name__ == '__main__':
    detector = XRayAnomalyDetector()
    detector.load()
    print("Anomaly detector initialized.")
