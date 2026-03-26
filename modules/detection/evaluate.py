from ultralytics import YOLO
import json, csv
from pathlib import Path
import os

def full_evaluation(model_path, data_yaml, output_dir='results/metrics'):
    # Resolve paths relative to project root
    if not os.path.isabs(model_path):
        # Get project root (2 levels up from modules/detection)
        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = project_root / model_path
        data_yaml = project_root / data_yaml
        output_dir = project_root / output_dir
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found: {model_path}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Script location: {Path(__file__).resolve()}")
        exit(1)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"Running evaluation on: {data_yaml}")
    metrics = model.val(data=str(data_yaml), split='test', save_json=True, plots=True)
    
    report = {
        'mAP_50':       round(float(metrics.box.map50), 4),
        'mAP_50_95':    round(float(metrics.box.map),   4),
        'precision':    round(float(metrics.box.mp),    4),
        'recall':       round(float(metrics.box.mr),    4),
        'f1':           round(2 * float(metrics.box.mp) * float(metrics.box.mr) /
                             (float(metrics.box.mp) + float(metrics.box.mr) + 1e-9), 4),
        'per_class_ap50': {
            name: round(float(ap), 4)
            for name, ap in zip(model.names.values(), metrics.box.ap50)
        }
    }
    
    # Save to JSON
    output_file = Path(output_dir) / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print('='*50)
    print('EVALUATION RESULTS')
    print('='*50)
    for k, v in report.items():
        if k != 'per_class_ap50':
            print(f'{k:20s}: {v}')
    print()
    print('Per-class AP@0.5:')
    for name, ap in report['per_class_ap50'].items():
        bar = '#' * int(ap * 30)
        print(f'  {name:12s}: {ap:.4f} |{bar}')
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return report

if __name__ == '__main__':
    full_evaluation('models/yolo/best.pt', 'data/data.yaml')
