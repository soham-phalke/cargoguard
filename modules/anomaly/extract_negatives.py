import shutil
from pathlib import Path
from tqdm import tqdm

def extract_clean_images(data_root, output_dir, max_images=5000):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for split in ['train', 'val']:
        lbl_dir = data_root / split / 'labels'
        img_dir = data_root / split / 'images'
        for lbl_file in tqdm(list(lbl_dir.glob('*.txt')), desc=split):
            content = lbl_file.read_text().strip()
            if content == '':  # empty label = no prohibited items
                for ext in ['.jpg', '.jpeg', '.png']:
                    img = img_dir / (lbl_file.stem + ext)
                    if img.exists():
                        shutil.copy(img, output_dir / img.name)
                        count += 1
                        break
            if count >= max_images: break
        if count >= max_images: break
    print(f'Extracted {count} clean images to {output_dir}')

if __name__ == '__main__':
    extract_clean_images(
        data_root='data/processed/merged',
        output_dir='data/processed/clean_images'
    )
