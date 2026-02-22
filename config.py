"""
config.py — Paths, constants, and thresholds.
All directories are relative to this file's location (the project root).
"""

import os

# ── Project root ────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ──────────────────────────────────────────────────
DATA_DIR         = os.path.join(ROOT, "data")
KINS_DIR         = os.path.join(DATA_DIR, "KINS")
ANNOT_DIR        = os.path.join(KINS_DIR, "annotations")
KITTI_DIR        = os.path.join(DATA_DIR, "kitti_extracted")
KITTI_TRAIN_IMGS = os.path.join(KITTI_DIR, "training", "image_2")
KITTI_TEST_IMGS  = os.path.join(KITTI_DIR, "testing", "image_2")

# ── AISFormer paths ─────────────────────────────────────────────
AISFORMER_DIR    = os.path.join(ROOT, "AISFormer")

# ── Output paths ────────────────────────────────────────────────
OUTPUT_DIR       = os.path.join(ROOT, "output")
GT_VIS_DIR       = os.path.join(OUTPUT_DIR, "01_ground_truth")
COMPARE_DIR      = os.path.join(OUTPUT_DIR, "02_comparisons")
CHARTS_DIR       = os.path.join(OUTPUT_DIR, "03_analysis_charts")
DEEPDIVE_DIR     = os.path.join(OUTPUT_DIR, "04_failure_deep_dive")
RESULTS_ZIP      = os.path.join(OUTPUT_DIR, "AmodalSeg_Results.zip")

# ── Download sources ────────────────────────────────────────────
KINS_GDRIVE_FOLDER = "1FuXz1Rrv5rrGG4n7KcQHVWKvSyr3Tkyo"
KITTI_URL          = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
COCO_WEIGHTS       = ("detectron2://COCO-InstanceSegmentation/"
                      "mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl")

# ── Analysis settings ───────────────────────────────────────────
N_GT_VIS       = 10      # Ground-truth visualisation images
N_TEST_IMAGES  = 15      # Images for baseline inference
SCORE_THRESH   = 0.5     # Detector confidence threshold
RANDOM_SEED    = 123     # Reproducibility

# ── Hardware thresholds ─────────────────────────────────────────
MIN_DISK_GB         = 20
MIN_VRAM_GB         = 2.0
RECOMMENDED_VRAM_GB = 4.0

# ── COCO class names (80 classes, id 0–79) ──────────────────────
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

DRIVING_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
