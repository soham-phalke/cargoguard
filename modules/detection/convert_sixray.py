import os, xml.etree.ElementTree as ET, shutil
from pathlib import Path
from tqdm import tqdm

CLASS_MAP = {
    'Folding_Knife':0, 'Straight_Knife':0,  # both → class 1 Knife
    'Scissor':4, 'Multi-tool_Knife':0,
    'Utility_Knife':0
}

def convert_xml_to_yolo(xml_path, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_MAP: continue
        cls_id = CLASS_MAP[name]
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        cx = ((x1+x2)/2) / img_w
        cy = ((y1+y2)/2) / img_h
        w  = (x2-x1) / img_w
        h  = (y2-y1) / img_h
        lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    return lines

def convert_opixray(opixray_root, output_root):
    opixray_root = Path(opixray_root)
    output_root  = Path(output_root)
    from PIL import Image
    for split in ['train', 'test']:
        img_dir = opixray_root / split / 'images'
        ann_dir = opixray_root / split / 'annotations'
        out_img = output_root / split / 'images'
        out_lbl = output_root / split / 'labels'
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for xml_file in tqdm(list(ann_dir.glob('*.xml')), desc=f'OPIXray {split}'):
            stem = xml_file.stem
            img_file = img_dir / (stem + '.jpg')
            if not img_file.exists(): continue
            img = Image.open(img_file)
            w, h = img.size
            lines = convert_xml_to_yolo(xml_file, w, h)
            shutil.copy(img_file, out_img / img_file.name)
            lbl_path = out_lbl / (stem + '.txt')
            lbl_path.write_text('\n'.join(lines))

if __name__ == '__main__':
    convert_opixray('data/raw/OPIXray', 'data/processed/opixray_yolo')
    print('OPIXray conversion complete!')
