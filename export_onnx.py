"""
Export YOLO model to ONNX format for faster CPU inference
"""
from ultralytics import YOLO
import os

print("=" * 60)
print("YOLO Model ONNX Export")
print("=" * 60)

model_path = 'models/yolo/best.pt'

if not os.path.exists(model_path):
    print(f"\n❌ ERROR: Model not found: {model_path}")
    exit(1)

print(f"\nLoading model: {model_path}")
model = YOLO(model_path)

print("\nExporting to ONNX format...")
print("  - Image size: 640x640")
print("  - OPSET version: 12")
print("  - Simplify: True")
print("\nThis may take 1-2 minutes...\n")

try:
    model.export(
        format='onnx',
        imgsz=640,
        opset=12,
        simplify=True
    )
    
    onnx_path = model_path.replace('.pt', '.onnx')
    
    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print("\n" + "=" * 60)
        print("✓ ONNX Export Complete!")
        print("=" * 60)
        print(f"\nExported model: {onnx_path}")
        print(f"File size: {size_mb:.2f} MB")
        print("\nBenefits:")
        print("  • 2-3x faster inference on CPU")
        print("  • Smaller model size")
        print("  • Cross-platform compatibility")
        print("\nUsage:")
        print("  model = YOLO('models/yolo/best.onnx')")
    else:
        print("\n⚠ Warning: ONNX file not found at expected location")
        
except Exception as e:
    print(f"\n❌ ERROR during export: {e}")
    exit(1)
