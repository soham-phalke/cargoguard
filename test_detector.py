"""
Test script for the complete Module 1 detection pipeline
"""
import os
import sys
from glob import glob

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from modules.detection.detector import CargoDetector

def main():
    print("=" * 60)
    print("Testing CargoDetector Pipeline")
    print("=" * 60)
    
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
        matches = glob(pattern)
        if matches:
            test_img = matches[0]
            break
    
    if not test_img:
        print("\n❌ ERROR: No test images found in data/processed/merged/")
        print("   Please ensure the dataset has been extracted/merged.")
        print("   Expected location: data/processed/merged/{train,val,test}/images/")
        print("\n   To extract the dataset, run:")
        print("   - Extract data/processed/merged.tar.gz to data/processed/merged/")
        return
    
    print(f"\nUsing test image: {test_img}")
    
    # Initialize detector
    print("\n[1/3] Loading YOLO model...")
    detector = CargoDetector('models/yolo/best.pt', conf_threshold=0.25)
    
    # Run detection
    print(f"\n[2/3] Running detection...")
    result = detector.run_detection(test_img)
    
    # Display results
    print("\n[3/3] Detection Results:")
    print("=" * 60)
    print(f"Detection Count: {result['detection_count']}")
    print(f"Detection Score: {result['det_score']}/100")
    print(f"Has Threat:      {result['has_threat']}")
    print(f"Inference Time:  {result['inference_ms']}ms")
    print(f"\nExplanation:\n  {result['explanation']}")
    
    if result['detections']:
        print(f"\nDetailed Detections:")
        print("-" * 60)
        for i, d in enumerate(result['detections'], 1):
            print(f"{i}. {d['class_name'].upper():12s} | Conf: {d['confidence_pct']:6s} | "
                  f"Risk: {d['item_risk']:3d} | Category: {d['category']}")
            print(f"   Box: {d['box']}")
            print(f"   {d['explanation']}")
            print()
    
    # Check annotated image
    if result['annotated_image']:
        img_size = len(result['annotated_image'])
        print(f"✓ Annotated image generated (base64, {img_size:,} chars)")
    
    print("\n" + "=" * 60)
    print("✓ Detection pipeline test complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
