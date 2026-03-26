import cv2, numpy as np, base64, time
from pathlib import Path
from ultralytics import YOLO
from modules.detection.gradcam import YOLOGradCAM

RISK_LEVELS = {
    'gun':     {'risk': 95, 'category': 'PROHIBITED',  'color': 'red'},
    'knife':   {'risk': 90, 'category': 'PROHIBITED',  'color': 'red'},
    'wrench':  {'risk': 40, 'category': 'RESTRICTED',  'color': 'amber'},
    'pliers':  {'risk': 35, 'category': 'RESTRICTED',  'color': 'amber'},
    'scissors':{'risk': 50, 'category': 'RESTRICTED',  'color': 'amber'},
    'hammer':    {'risk': 25, 'category': 'TOOL',        'color': 'blue'},
    'powerbank':  {'risk': 45, 'category': 'RESTRICTED',  'color': 'amber'},
    'baton':      {'risk': 70, 'category': 'PROHIBITED',  'color': 'red'},
    'bullet':     {'risk': 95, 'category': 'PROHIBITED',  'color': 'red'},
    'sprayer':    {'risk': 40, 'category': 'RESTRICTED',  'color': 'amber'},
    'handcuffs':  {'risk': 30, 'category': 'RESTRICTED',  'color': 'amber'},
    'lighter':    {'risk': 20, 'category': 'TOOL',        'color': 'blue'},
}

EXPLANATIONS = {
    'gun':     'Firearm detected — prohibited item. Manual inspection required.',
    'knife':   'Bladed weapon detected — prohibited. Requires immediate review.',
    'wrench':  'Tool detected — context-dependent. Verify declared cargo type.',
    'pliers':  'Pliers detected — tool category. Verify against manifest.',
    'scissors':'Scissors detected — restricted item. Confirm declared use.',
    'hammer':    'Hammer detected — standard tool. Low risk if cargo matches.',
    'powerbank':  'Powerbank detected — check for hidden compartments or modified cells.',
    'baton':      'Baton/club detected — prohibited item. Manual inspection required.',
    'bullet':     'Ammunition detected — prohibited. Immediate escalation required.',
    'sprayer':    'Spray canister detected — verify contents and declared cargo.',
    'handcuffs':  'Restraint device detected — verify authorization and manifest.',
    'lighter':    'Lighter detected — flammable item. Confirm cargo type.',
}

class CargoDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, device: str = 'cpu'):
        self.model = YOLO(model_path)
        self.gradcam = YOLOGradCAM(model_path, device=device)
        self.conf = conf_threshold
        self.iou  = iou_threshold
        self.device = device
        print(f'CargoDetector loaded. Model: {model_path}')
    
    def run_detection(self, image_path: str) -> dict:
        """
        Run full detection pipeline on one image.
        Returns standardised dict for the risk engine.
        """
        start_time = time.time()
        
        # ── 1. Run YOLO inference ──────────────────────────
        results = self.model.predict(
            source=image_path,
            conf=self.conf,
            iou=self.iou,
            imgsz=640,
            device=self.device,
            verbose=False
        )
        
        inference_ms = int((time.time() - start_time) * 1000)
        
        # ── 2. Parse detections ───────────────────────────
        detections = []
        max_risk = 0
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls.item())
                cls_name = self.model.names[cls_idx]
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                risk_info = RISK_LEVELS.get(cls_name, {'risk':20,'category':'UNKNOWN','color':'gray'})
                item_risk = int(risk_info['risk'] * conf)
                max_risk = max(max_risk, item_risk)
                
                detections.append({
                    'class_name':  cls_name,
                    'class_idx':   cls_idx,
                    'confidence':  round(conf, 4),
                    'confidence_pct': f'{conf:.0%}',
                    'box':         [round(v) for v in xyxy],
                    'category':    risk_info['category'],
                    'color':       risk_info['color'],
                    'item_risk':   item_risk,
                    'explanation': EXPLANATIONS.get(cls_name, ''),
                })
        
        # Sort by confidence descending
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        
        # ── 3. Generate annotated image with Grad-CAM ─────
        annotated = self.gradcam.annotate_image(image_path, detections)
        annotated_b64 = self._img_to_b64(annotated)
        
        # ── 4. Build explanation ──────────────────────────
        if detections:
            top = detections[0]
            explanation = f"{top['category']}: {top['class_name'].capitalize()} detected at {top['confidence_pct']} confidence. {top['explanation']}"
        else:
            explanation = 'No known prohibited items detected by object detection module.'
            max_risk = 0
        
        return {
            'detections':      detections,
            'detection_count': len(detections),
            'det_score':       max_risk,          # 0-100, used by risk engine
            'annotated_image': annotated_b64,     # base64 PNG
            'explanation':     explanation,
            'inference_ms':    inference_ms,
            'has_threat':      len(detections) > 0
        }
    
    def _img_to_b64(self, img_rgb: np.ndarray) -> str:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf).decode('utf-8')


# ── Quick test ──────────────────────────────────────────────
if __name__ == '__main__':
    detector = CargoDetector('models/yolo/best.pt')
    result = detector.run_detection('data/processed/merged/test/images/P010001.jpg')
    print(f'Detections: {result["detection_count"]}')
    print(f'Det score:  {result["det_score"]}')
    print(f'Inference:  {result["inference_ms"]}ms')
    for d in result['detections']:
        print(f'  {d["class_name"]:12s} {d["confidence_pct"]:6s} [{d["category"]}]')
