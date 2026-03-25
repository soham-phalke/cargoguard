# CargoGuard SENTINEL

AI-assisted cargo X-ray screening project for prohibited item detection and future multi-stage risk analysis.

This repository currently includes:
- A YOLO-based detection training pipeline for X-ray images.
- Dataset conversion and merge utilities.
- Starter modules for anomaly detection and manifest mismatch verification.
- A project structure ready for backend and frontend integration.

## 1) Project Objective

CargoGuard SENTINEL aims to support security screening teams by:
- Detecting dangerous or restricted objects in X-ray images.
- Assigning actionable risk categories.
- (Planned) combining supervised detection with anomaly detection and manifest verification.

## 2) Repository Structure

Main directories:
- app/: future application layer
	- backend/
	- frontend/
	- static/
- data/: dataset configs and local data folders
- models/: model storage (local, ignored in Git)
- modules/: core Python modules
	- detection/: dataset preparation and YOLO training scripts
	- anomaly/: anomaly detector scaffolding
	- verification/: mismatch verification logic
- notebooks/: Kaggle/local experimentation notebooks
- results/: outputs/metrics/visualizations (local, ignored in Git)

Note:
- Empty app folders are tracked using .gitkeep files.
- Large datasets and model artifacts are intentionally ignored in Git for repository size control.

## 3) Detection Pipeline (Current Core)

The main implemented flow is:

1. Convert raw annotations to YOLO format
	- PIDray conversion script: modules/detection/convert_pidray.py
	- Additional XML conversion utility: modules/detection/convert_sixray.py

2. Merge and split datasets
	- Script: modules/detection/merge_datasets.py
	- Output format: data/processed/merged/{train,val,test}/{images,labels}

3. Validate class/split distribution
	- Script: modules/detection/dataset_stats.py

4. Train YOLOv8 model
	- Local training script: modules/detection/train_local.py
	- Kaggle version: notebooks/kaggle_training.py

5. Use trained weights
	- Typical output: runs/train/cargoguard_pidray/weights/best.pt

## 4) Classes and Risk Categories

The project supports a 12-class taxonomy (PIDray mapping):
- Baton
- Pliers
- Hammer
- Powerbank
- Scissors
- Wrench
- Gun
- Bullet
- Sprayer
- HandCuffs
- Knife
- Lighter

Risk grouping used in the project documentation:
- PROHIBITED (high risk): Gun, Knife, Baton, Bullet
- RESTRICTED (medium risk): Pliers, Scissors, Wrench, Hammer, HandCuffs
- REGULATED (low-to-medium risk): Powerbank, Sprayer, Lighter

Reference file: modules/detection/class_structure.py

## 5) Quick Start

### Prerequisites
- Python 3.10+ recommended
- pip
- GPU recommended for training (CUDA)

### Environment setup (Windows PowerShell)

1. Create venv
	 python -m venv venv

2. Activate
	 .\\venv\\Scripts\\Activate.ps1

3. Install dependencies
	 pip install -r requirements.txt

## 6) Data Preparation

Important:
- Raw and processed dataset folders are large and are not pushed to GitHub by default.
- Teammates need a separate dataset download/copy step.

### A) Convert PIDray to YOLO

Run:
python modules/detection/convert_pidray.py

Default output:
- data/processed/pidray_yolo

### B) Merge datasets for training

Run:
python modules/detection/merge_datasets.py

Default output:
- data/processed/merged

### C) Check class distribution

Run:
python modules/detection/dataset_stats.py data/processed/merged

## 7) Model Training

### Local training

Run:
python modules/detection/train_local.py --data data/data.yaml --epochs 50 --batch 16

Notes:
- If you do not pass --data, the script defaults to data/processed/pidray_yolo/data.yaml.
- Update data/data.yaml path if your merged dataset lives at a different location.

### Kaggle training

Use:
- notebooks/kaggle_training.py
- notebooks/train_yolov8_kaggle.ipynb

## 8) Current Module Status

- detection module: functional for conversion, merging, stats, and YOLO training.
- anomaly module: scaffold present (placeholder behavior if anomalib is unavailable).
- verification module: mismatch verification logic present, depends on embedding builder assets.
- app layer: folder structure present; implementation pending.

## 9) Configuration Notes

File: data/data.yaml
- Contains train/val/test paths and class names.
- Keep paths valid for your local machine or cloud runtime.
- If sharing with teammates, use relative paths when possible.

## 10) Recommended Team Workflow

1. Clone repository.
2. Set up Python environment and install requirements.
3. Download/copy dataset into expected local directories.
4. Run conversion and merge scripts if needed.
5. Verify stats with dataset_stats.py.
6. Train using train_local.py or Kaggle notebook.
7. Save best weights and report metrics in results/metrics.

## 11) PPT-Ready Summary

Use this section directly in presentation slides.

Slide 1: Problem Statement
- Manual cargo X-ray screening is time-consuming and error-prone.
- Security teams need faster and consistent prohibited-item detection.

Slide 2: Proposed Solution
- CargoGuard SENTINEL uses computer vision to detect suspicious objects in X-rays.
- Pipeline supports data conversion, YOLO training, and risk-oriented class mapping.

Slide 3: Technical Architecture
- Input: raw X-ray datasets (PIDray/SIXray).
- Processing: annotation conversion + merged YOLO dataset creation.
- Model: YOLOv8 training and evaluation.
- Extensions: anomaly detection and manifest mismatch verification modules.

Slide 4: Current Progress
- Dataset conversion scripts implemented.
- Merging and split automation implemented.
- Local and Kaggle training scripts implemented.
- Modular repository prepared for backend/frontend integration.

Slide 5: Expected Impact
- Faster screening throughput.
- Better consistency in prohibited item identification.
- Scalable foundation for smart, multi-stage cargo risk assessment.

## 12) Limitations and Next Steps

Current limitations:
- Datasets and trained weights are not versioned in Git by default.
- App backend/frontend is scaffolded but not fully built.

Next steps:
- Add a dataset distribution strategy (cloud bucket, release asset, or Git LFS).
- Add model evaluation report templates and benchmark tracking.
- Integrate trained model into backend inference API and frontend dashboard.

## 13) Key Files

- modules/detection/convert_pidray.py
- modules/detection/merge_datasets.py
- modules/detection/dataset_stats.py
- modules/detection/train_local.py
- modules/detection/class_structure.py
- modules/anomaly/anomaly_detector.py
- modules/verification/mismatch.py
- notebooks/kaggle_training.py
- data/data.yaml

---

For contributors:
- Keep code changes modular by folder (detection, anomaly, verification, app).
- Avoid committing raw datasets, model binaries, and generated outputs unless explicitly required.
