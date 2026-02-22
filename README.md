# Amodal Instance Segmentation Analysis Tool
## AISFormer on KINS — Windows Desktop Application

A standalone Windows application that reproduces the Colab-based amodal
segmentation analysis pipeline locally. It downloads the KINS dataset and KITTI
images, runs a COCO-pretrained Faster R-CNN baseline, compares bounding-box
detections against amodal ground-truth masks, and generates quantitative charts
and failure-case visualizations — all from a single GUI window.

---

## Quick Start

### 1. Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 64-bit | Windows 11 |
| Python | 3.10 | 3.10–3.11 |
| GPU | NVIDIA with 2 GB VRAM | NVIDIA with 4+ GB VRAM |
| CUDA Toolkit | 11.8 or 12.1 | 12.1 |
| Disk space | 20 GB free | 30 GB free |
| RAM | 8 GB | 16 GB |

### 2. Create a Virtual Environment

Open **PowerShell** or **Command Prompt** and run:

```powershell
cd C:\path\to\this\folder
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch with CUDA

Go to https://pytorch.org/get-started/locally/ and pick your CUDA version.
Example for CUDA 12.1:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Remaining Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the Application

```powershell
python app.py
```

The GUI will open, show your hardware info, and let you start the full pipeline
with one click. All output images and charts are saved to the `output/` folder.

---

## What It Does

| Step | Description | Output |
|---|---|---|
| 1. Setup | Clones AISFormer, patches Detectron2 compatibility | Ready to run |
| 2. Download | KINS annotations + KITTI images (~12 GB) | `data/` folder |
| 3. Ground Truth | Visualizes amodal vs visible vs occluded masks | `output/01_ground_truth/` |
| 4. Baseline | Runs Faster R-CNN inference on 15 test images | Detection results |
| 5. Comparison | Side-by-side: detector boxes vs amodal GT | `output/02_comparisons/` |
| 6. Quantitative | Occlusion stats, recall curves, coverage charts | `output/03_analysis_charts/` |
| 7. Deep Dive | Cherry-picked failure cases under heavy occlusion | `output/04_failure_deep_dive/` |
| 8. AISFormer | Published results comparison charts | `output/03_analysis_charts/` |
| 9. Package | Zips everything into `output/AmodalSeg_Results.zip` | Ready to download |

---

## Project Structure

```
amodal_seg_app/
├── app.py              # Main GUI application (run this)
├── pipeline.py         # All analysis steps
├── config.py           # Paths, constants, thresholds
├── requirements.txt    # pip dependencies (except PyTorch)
├── README.md           # This file
├── data/               # Created at runtime (datasets go here)
└── output/             # Created at runtime (all results here)
```

---

## References

- **AISFormer**: Tran et al., "AISFormer: Amodal Instance Segmentation with
  Transformer," BMVC 2022. https://arxiv.org/abs/2210.06323
- **KINS Dataset**: Qi et al., "Amodal Instance Segmentation with KINS Dataset,"
  CVPR 2019.
- **Detectron2**: Wu et al., Facebook AI Research, 2019.
- **KITTI**: Geiger et al., "Are we ready for Autonomous Driving?" CVPR 2012.
