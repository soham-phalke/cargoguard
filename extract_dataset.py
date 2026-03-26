"""
Extract the merged dataset from tar.gz archive
"""
import tarfile
import os
from pathlib import Path

print("=" * 60)
print("Extracting Merged Dataset")
print("=" * 60)

archive_path = "data/processed/merged.tar.gz"
extract_to = "data/processed"

if not os.path.exists(archive_path):
    print(f"\n❌ ERROR: Archive not found: {archive_path}")
    exit(1)

print(f"\nArchive: {archive_path}")
print(f"Extract to: {extract_to}")
print(f"\nExtracting... (this may take a minute)")

try:
    with tarfile.open(archive_path, 'r:gz') as tar:
        # Get total members for progress
        members = tar.getmembers()
        total = len(members)
        
        # Extract with progress
        for i, member in enumerate(members, 1):
            tar.extract(member, path=extract_to)
            if i % 100 == 0 or i == total:
                print(f"  Extracted {i}/{total} files ({i*100//total}%)")
        
    print(f"\n✓ Successfully extracted {total} files!")
    
    # Verify extracted structure
    merged_dir = os.path.join(extract_to, "merged")
    if os.path.exists(merged_dir):
        print(f"\n✓ Dataset structure verified:")
        for split in ['train', 'val', 'test']:
            img_dir = os.path.join(merged_dir, split, 'images')
            if os.path.exists(img_dir):
                img_count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
                print(f"  {split:5s}: {img_count:4d} images")
    
    print("\n" + "=" * 60)
    print("✓ Dataset extraction complete!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python test_detector.py")
    print("  python modules/detection/gradcam.py")

except Exception as e:
    print(f"\n❌ ERROR during extraction: {e}")
    exit(1)
