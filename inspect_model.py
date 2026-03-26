from ultralytics import YOLO

model = YOLO('models/yolo/best.pt')

# Print model layers to find correct target
for i, (name, module) in enumerate(model.model.named_modules()):
    if 'Conv' in type(module).__name__:
        print(f'Layer {i}: {name} — {type(module).__name__}')

# The correct target is the last Conv2d before the detection head
# Usually: model.model.model[-2] for YOLOv8
