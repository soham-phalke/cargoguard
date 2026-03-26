"""
Compare PyTorch vs ONNX inference speed
"""
import time
from glob import glob
from modules.detection.detector import CargoDetector

print("=" * 70)
print("PYTORCH vs ONNX PERFORMANCE COMPARISON")
print("=" * 70)

# Find test image
test_img = glob('data/processed/merged/test/images/*.png')[0]
print(f"\nTest image: {test_img}\n")

# Test PyTorch model
print("[1/2] Testing PyTorch model...")
detector_pt = CargoDetector('models/yolo/best.pt', conf_threshold=0.25)
times_pt = []
for i in range(5):
    result = detector_pt.run_detection(test_img)
    times_pt.append(result['inference_ms'])
    print(f"  Run {i+1}: {result['inference_ms']}ms")

avg_pt = sum(times_pt) / len(times_pt)
print(f"  → Average: {avg_pt:.1f}ms\n")

# Test ONNX model
print("[2/2] Testing ONNX model...")
detector_onnx = CargoDetector('models/yolo/best.onnx', conf_threshold=0.25)
times_onnx = []
for i in range(5):
    result = detector_onnx.run_detection(test_img)
    times_onnx.append(result['inference_ms'])
    print(f"  Run {i+1}: {result['inference_ms']}ms")

avg_onnx = sum(times_onnx) / len(times_onnx)
print(f"  → Average: {avg_onnx:.1f}ms\n")

# Calculate speedup
speedup = avg_pt / avg_onnx
improvement = ((avg_pt - avg_onnx) / avg_pt) * 100

print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"PyTorch:  {avg_pt:.1f}ms")
print(f"ONNX:     {avg_onnx:.1f}ms")
print(f"Speedup:  {speedup:.2f}x faster")
print(f"Improved: {improvement:.1f}% faster")
print("=" * 70)

if speedup > 1.5:
    print("\n✓ Significant performance improvement! Use ONNX for production.")
elif speedup > 1.0:
    print("\n✓ Moderate improvement. ONNX is faster.")
else:
    print("\n⚠ No significant improvement. PyTorch may be better for this hardware.")
