"""
Class Structure Documentation - CargoGuard SENTINEL
Understanding prohibited item classes across datasets.
"""

# ============================================================
# DATASET COMPARISON
# ============================================================

# SIXray v3 Dataset (5 classes) - Currently Using
SIXRAY_CLASSES = {
    0: 'Gun',
    1: 'Knife',
    2: 'Pliers',
    3: 'Scissors',
    4: 'Wrench',
}

# PIDray Dataset (12 classes) - Alternative/Additional Dataset
PIDRAY_CLASSES = {
    1:  'Baton',
    2:  'Pliers',
    3:  'Hammer',
    4:  'Powerbank',
    5:  'Scissors',
    6:  'Wrench',
    7:  'Gun',
    8:  'Bullet',
    9:  'Sprayer',
    10: 'HandCuffs',
    11: 'Knife',
    12: 'Lighter',
}

# ============================================================
# RISK CLASSIFICATION
# ============================================================

RISK_CATEGORIES = {
    'PROHIBITED': {
        'level': 'RED',
        'risk_score': 90-95,
        'action': 'Immediate manual inspection required',
        'classes': ['Gun', 'Knife', 'Bullet', 'Baton']
    },
    'RESTRICTED': {
        'level': 'AMBER',
        'risk_score': 40-60,
        'action': 'Secondary review recommended',
        'classes': ['Scissors', 'Wrench', 'Pliers', 'Hammer', 'HandCuffs']
    },
    'REGULATED': {
        'level': 'YELLOW',
        'risk_score': 30-50,
        'action': 'Verify against manifest',
        'classes': ['Powerbank', 'Sprayer', 'Lighter']
    },
}

# ============================================================
# CLASS MAPPING FOR UNIFIED MODEL (Optional)
# ============================================================
# If training on both datasets, map to unified classes

UNIFIED_MAPPING = {
    # Weapons - Class 0
    'Gun': 0,
    'Bullet': 0,
    'Knife': 0,
    'Baton': 0,

    # Sharp Tools - Class 1
    'Scissors': 1,
    'Pliers': 1,

    # Heavy Tools - Class 2
    'Wrench': 2,
    'Hammer': 2,

    # Electronics/Regulated - Class 3
    'Powerbank': 3,
    'Sprayer': 3,
    'Lighter': 3,

    # Restraints - Class 4
    'HandCuffs': 4,
}

# ============================================================
# PIDRAY TO YOLO CONVERSION MAPPING
# ============================================================
# YOLO expects 0-indexed classes
# PIDray JSON uses 1-indexed classes

PIDRAY_TO_YOLO = {
    1:  0,   # Baton
    2:  1,   # Pliers
    3:  2,   # Hammer
    4:  3,   # Powerbank
    5:  4,   # Scissors
    6:  5,   # Wrench
    7:  6,   # Gun
    8:  7,   # Bullet
    9:  8,   # Sprayer
    10: 9,   # HandCuffs
    11: 10,  # Knife
    12: 11,  # Lighter
}

YOLO_CLASS_NAMES = [
    'Baton',      # 0
    'Pliers',     # 1
    'Hammer',     # 2
    'Powerbank',  # 3
    'Scissors',   # 4
    'Wrench',     # 5
    'Gun',        # 6
    'Bullet',     # 7
    'Sprayer',    # 8
    'HandCuffs',  # 9
    'Knife',      # 10
    'Lighter',    # 11
]


# ============================================================
# DATASET STATISTICS
# ============================================================

DATASET_INFO = {
    'SIXray_v3': {
        'total_images': 14131,
        'num_classes': 5,
        'format': 'YOLO',
        'splits': {
            'train': 11638,
            'valid': 1662,
            'test': 831
        },
        'source': 'Roboflow (Kaggle)',
        'license': 'CC BY 4.0'
    },
    'PIDray': {
        'total_images': 29457,
        'num_classes': 12,
        'format': 'COCO JSON',
        'splits': {
            'train': 23732,
            'easy': 2831,
            'hard': 1466,
            'hidden': 1428
        },
        'source': 'GitHub (bywang2018)',
        'license': 'Academic Use'
    }
}


# ============================================================
# RECOMMENDED APPROACH
# ============================================================

RECOMMENDATION = """
RECOMMENDED STRATEGY:

Option A - Quick Start (SIXray only):
  [+] Use SIXray v3 (already in YOLO format)
  [+] 5 classes, 14K images
  [+] Ready to train immediately
  [+] Expected mAP@0.5: 82-86%

Option B - Best Performance (PIDray + SIXray):
  [+] Convert PIDray to YOLO format
  [+] Merge with SIXray
  [+] 12 classes, 43K+ images
  [+] More comprehensive threat detection
  [+] Expected mAP@0.5: 85-90%
  [!] Requires data conversion (~30 min)

Current Setup: Using Option A (SIXray only)
To switch to Option B: Run convert_pidray.py
"""

if __name__ == '__main__':
    print("=" * 60)
    print("CARGOGUARD SENTINEL - CLASS STRUCTURE")
    print("=" * 60)
    print()

    print("SIXray v3 Classes (5):")
    for id, name in SIXRAY_CLASSES.items():
        print(f"  {id}: {name}")
    print()

    print("PIDray Classes (12):")
    for id, name in PIDRAY_CLASSES.items():
        print(f"  {id}: {name}")
    print()

    print(RECOMMENDATION)
