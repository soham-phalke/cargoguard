"""
Quick detector test with saved visualization
"""
import os
import sys
import cv2
import base64
from glob import glob

sys.path.insert(0, os.path.dirname(__file__))
from modules.detection.detector import CargoDetector

# Find test images
test_images = glob('data/processed/merged/test/images/*.png')[:3]  # Test first 3

if not test_images:
    print("No test images found!")
    exit(1)

print("=" * 70)
print("CARGOGUARD DETECTOR TEST")
print("=" * 70)

# Initialize detector
detector = CargoDetector('models/yolo/best.pt', conf_threshold=0.25)

# Create output directory
os.makedirs('results/detections', exist_ok=True)

for i, img_path in enumerate(test_images, 1):
    print(f"\n[TEST {i}/{len(test_images)}] {os.path.basename(img_path)}")
    print("-" * 70)
    
    # Run detection
    result = detector.run_detection(img_path)
    
    # Display results
    print(f"✓ Detections: {result['detection_count']}")
    print(f"✓ Risk Score: {result['det_score']}/100")
    print(f"✓ Has Threat: {result['has_threat']}")
    print(f"✓ Inference:  {result['inference_ms']}ms")
    
    if result['detections']:
        print(f"\nDetected Items:")
        for d in result['detections']:
            print(f"  • {d['class_name'].upper():12s} - {d['confidence_pct']} confidence "
                  f"[{d['category']}] Risk: {d['item_risk']}")
    else:
        print("  • No threats detected")
    
    # Save annotated image
    img_data = base64.b64decode(result['annotated_image'])
    output_path = f"results/detections/test_{i}_annotated.jpg"
    with open(output_path, 'wb') as f:
        f.write(img_data)
    print(f"\n✓ Saved: {output_path}")

print("\n" + "=" * 70)
print("✓ DETECTOR TEST COMPLETE!")
print("=" * 70)
print(f"\nAnnotated images saved to: results/detections/")
