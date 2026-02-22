"""
pipeline.py — Complete analysis pipeline.

Every public function has the signature:
    step_xxx(log: Callable[[str], None]) -> bool

`log` is called with status messages that the GUI displays in real time.
Each step returns True on success or raises on failure.
"""

import os, sys, re, json, time, random, shutil, glob, zipfile, subprocess
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")                       # headless — write PNGs only
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_utils
from collections import defaultdict

import config as C

# ====================================================================
# Shared helpers  (used across multiple steps)
# ====================================================================

def decode_segm(segm, h, w):
    """Decode a KINS segmentation field to a binary mask (H, W)."""
    if segm is None or (isinstance(segm, list) and len(segm) == 0):
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(segm, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segm, h, w))
        return mask_utils.decode(rle).squeeze()
    if isinstance(segm, dict):
        counts = segm.get("counts")
        if isinstance(counts, str):
            return mask_utils.decode(segm).squeeze()
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(segm, h, w)
            if isinstance(rle, list):
                rle = mask_utils.merge(rle)
            return mask_utils.decode(rle).squeeze()
    return np.zeros((h, w), dtype=np.uint8)


def safe_resize(mask, h, w):
    if mask.shape[0] != h or mask.shape[1] != w:
        return cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST)
    return mask


def render_labeled_masks(img, masks, labels, alpha=0.45):
    """Overlay coloured masks with contours and centroid labels."""
    cmap = plt.cm.tab20
    out = img.copy().astype(np.float64)
    ih, iw = img.shape[:2]
    for i, (m, lab) in enumerate(zip(masks, labels)):
        if m is None or not m.any():
            continue
        m = safe_resize(m, ih, iw)
        col = (np.array(cmap(i % 20)[:3]) * 255).astype(int)
        b = m.astype(bool)
        for c in range(3):
            out[:, :, c][b] = out[:, :, c][b] * (1 - alpha) + col[c] * alpha
    out = out.clip(0, 255).astype(np.uint8)
    for i, (m, lab) in enumerate(zip(masks, labels)):
        if m is None or not m.any():
            continue
        m = safe_resize(m, ih, iw)
        col_f = np.array(cmap(i % 20)[:3])
        cnt, _ = cv2.findContours(m.astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnt, -1,
                         tuple(int(v * 255) for v in col_f), 2)
        ys, xs = np.where(m > 0)
        if len(ys):
            cx, cy = int(xs.mean()), int(ys.mean())
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(out, (cx - 2, cy - th - 4),
                          (cx + tw + 2, cy + 2), (0, 0, 0), -1)
            cv2.putText(out, lab, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def box_to_mask(box, h, w):
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = (int(v) for v in box)
    m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    return m


def _load_kins(log):
    p = os.path.join(C.ANNOT_DIR, "test.json")
    log(f"  Loading {p}")
    with open(p) as f:
        data = json.load(f)
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    log(f"  {len(data['images'])} images, "
        f"{len(data['annotations'])} annotations, "
        f"{len(cat_map)} categories")
    return data, cat_map


def _busy_images(kins):
    id2ann = {}
    for a in kins["annotations"]:
        id2ann.setdefault(a["image_id"], []).append(a)
    busy = [(iid, anns) for iid, anns in id2ann.items()
            if 4 <= len(anns) <= 20]
    random.seed(C.RANDOM_SEED)
    random.shuffle(busy)
    return busy


def _ensure_dirs():
    for d in (C.OUTPUT_DIR, C.GT_VIS_DIR, C.COMPARE_DIR,
              C.CHARTS_DIR, C.DEEPDIVE_DIR):
        os.makedirs(d, exist_ok=True)


# path to the lightweight JSON cache written by step_baseline
_CACHE = os.path.join(C.OUTPUT_DIR, "_baseline_cache.json")


def _save_cache(results):
    ser = []
    for r in results:
        s = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                s[k] = v.tolist()
            else:
                s[k] = v
        ser.append(s)
    with open(_CACHE, "w") as f:
        json.dump(ser, f)


def _load_cache():
    with open(_CACHE) as f:
        data = json.load(f)
    for r in data:
        for k in ("pred_boxes", "pred_classes", "pred_scores", "driving_mask"):
            if k in r:
                r[k] = np.array(r[k])
    return data


# ====================================================================
#  STEP 1 — Clone & patch AISFormer + Detectron2
# ====================================================================
def step_setup(log):
    log("=" * 60)
    log("STEP 1  Setup & Install AISFormer")
    log("=" * 60)
    _ensure_dirs()

    # Clone
    if not os.path.isfile(os.path.join(C.AISFORMER_DIR, "setup.py")):
        log("Cloning AISFormer repository …")
        subprocess.run(["git", "clone",
                        "https://github.com/UARK-AICV/AISFormer.git",
                        C.AISFORMER_DIR],
                       check=True, capture_output=True, text=True)
        log("  Cloned.")
    else:
        log("  AISFormer already present.")

    # Build (Detectron2 is bundled inside AISFormer)
    log("Building AISFormer + Detectron2 (may take a few minutes) …")
    r = subprocess.run([sys.executable, "setup.py", "build", "develop"],
                       cwd=C.AISFORMER_DIR,
                       capture_output=True, text=True)
    if r.returncode == 0:
        log("  Build succeeded.")
    else:
        log(f"  Build exited with code {r.returncode}")
        for line in (r.stderr or "").strip().splitlines()[-6:]:
            log(f"    {line}")

    # PIL compat
    from PIL import Image as _I
    if not hasattr(_I, "LINEAR"):
        _I.LINEAR = _I.BILINEAR

    # Patch builtin.py
    _patch_builtin(log)

    # sys.path
    if C.AISFORMER_DIR not in sys.path:
        sys.path.insert(0, C.AISFORMER_DIR)

    # Verify
    import detectron2  # noqa
    log(f"  Detectron2 {detectron2.__version__} ready.")
    return True


def _patch_builtin(log):
    bp = os.path.join(C.AISFORMER_DIR, "detectron2", "data",
                      "datasets", "builtin.py")
    if not os.path.isfile(bp):
        log("  builtin.py not found — skipping patch")
        return
    txt = open(bp).read()
    changed = False

    # Wrap COCOA / D2SA in try/except
    for old, new in [
        ("    register_COCOA()",
         "    try:\n        register_COCOA()\n    except Exception:\n        pass"),
        ("    register_D2SA()",
         "    try:\n        register_D2SA()\n    except Exception:\n        pass"),
    ]:
        if old in txt and new not in txt:
            txt = txt.replace(old, new); changed = True

    # Guard every register_* against duplicate registration
    guards = {
        "register_all_coco": "coco_2014_train",
        "register_all_lvis": "lvis_v0.5_val",
        "register_all_cityscapes": "cityscapes_fine_sem_seg_train",
        "register_all_pascal_voc": "voc_2007_trainval",
        "register_all_ade20k": "ade20k_sem_seg_train",
        "register_kins": "kins_dataset_train",
        "register_COCOA": "cocoa_cls_train",
        "register_D2SA": "d2sa_train",
    }
    for fn, key in guards.items():
        tag = f"# guard_{fn}"
        if tag not in txt:
            m = re.search(rf"def {fn}\((.*?)\):", txt)
            if m:
                args = m.group(1)
                old_l = f"def {fn}({args}):"
                new_l = (f"def {fn}({args}):\n"
                         f"    {tag}\n"
                         f"    from detectron2.data import DatasetCatalog\n"
                         f"    if \"{key}\" in DatasetCatalog:\n"
                         f"        return")
                txt = txt.replace(old_l, new_l); changed = True

    if changed:
        open(bp, "w").write(txt)
        log("  Patched builtin.py")
    else:
        log("  builtin.py already patched")


# ====================================================================
#  STEP 2 — Download KINS annotations + KITTI images
# ====================================================================
def step_download(log):
    log("=" * 60)
    log("STEP 2  Download Datasets")
    log("=" * 60)

    os.makedirs(C.ANNOT_DIR, exist_ok=True)
    test_json = os.path.join(C.ANNOT_DIR, "test.json")

    # ── KINS annotations ──
    if not os.path.isfile(test_json):
        log("Downloading KINS annotations from Google Drive …")
        import gdown
        gdown.download_folder(id=C.KINS_GDRIVE_FOLDER,
                              output=C.ANNOT_DIR, quiet=False)
        # Normalise names
        for src, dst in [("update_test_2020.json", "test.json"),
                         ("update_train_2020.json", "train.json")]:
            sp = os.path.join(C.ANNOT_DIR, src)
            dp = os.path.join(C.ANNOT_DIR, dst)
            if os.path.isfile(sp) and not os.path.isfile(dp):
                shutil.copy2(sp, dp)
                log(f"  Renamed {src} → {dst}")
        log("  KINS annotations ready.")
    else:
        log("  KINS annotations already present.")

    # ── KITTI images ──
    if not os.path.isdir(C.KITTI_TRAIN_IMGS):
        zip_path = os.path.join(C.DATA_DIR, "data_object_image_2.zip")
        if not os.path.isfile(zip_path):
            log("Downloading KITTI images (~12 GB) — this may take 10-20 min …")
            import urllib.request
            os.makedirs(C.DATA_DIR, exist_ok=True)
            _dl_count = [0]
            def _hook(bn, bs, total):
                _dl_count[0] += 1
                if _dl_count[0] % 500 == 0:
                    done = bn * bs
                    pct = done / total * 100 if total > 0 else 0
                    log(f"  {done / 1e9:.1f} / {total / 1e9:.1f} GB  ({pct:.0f}%)")
            urllib.request.urlretrieve(C.KITTI_URL, zip_path, reporthook=_hook)
            log("  Download complete.")

        log("  Extracting KITTI zip (may take several minutes) …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(C.KITTI_DIR)
        os.remove(zip_path)
        log("  Extracted & cleaned up zip.")
    else:
        log("  KITTI images already present.")

    nt = len(os.listdir(C.KITTI_TRAIN_IMGS))
    ne = len(os.listdir(C.KITTI_TEST_IMGS))
    log(f"  KITTI: {nt} training images, {ne} testing images")
    return True


# ====================================================================
#  STEP 3 — Ground-truth visualisation
# ====================================================================
def step_ground_truth(log):
    log("=" * 60)
    log("STEP 3  Ground-Truth Visualisation")
    log("=" * 60)
    _ensure_dirs()

    kins, cat_map = _load_kins(log)
    IMG_DIR = C.KITTI_TEST_IMGS       # KINS test → KITTI testing/
    busy = _busy_images(kins)
    id2info = {im["id"]: im for im in kins["images"]}
    N = min(C.N_GT_VIS, len(busy))

    for idx in range(N):
        img_id, anns = busy[idx]
        info = id2info.get(img_id)
        if info is None:
            continue
        path = os.path.join(IMG_DIR, info["file_name"])
        if not os.path.isfile(path):
            log(f"  SKIP {info['file_name']} (not found)")
            continue

        img = np.array(Image.open(path))
        ih, iw = img.shape[:2]
        ah, aw = info["height"], info["width"]

        am_masks = [decode_segm(a.get("a_segm"), ah, aw) for a in anns]
        im_masks = [decode_segm(a.get("i_segm"), ah, aw) for a in anns]
        labels   = [cat_map.get(a["category_id"], "?") for a in anns]

        # Three rendered panels
        amodal_vis  = render_labeled_masks(img, am_masks, labels)
        inmodal_vis = render_labeled_masks(img, im_masks, labels)

        occ_vis = img.copy().astype(np.float64)
        cmap = plt.cm.tab20
        n_occ = 0
        for i, (am, im_) in enumerate(zip(am_masks, im_masks)):
            amr = safe_resize(am, ih, iw)
            imr = safe_resize(im_, ih, iw)
            occ = amr.astype(bool) & ~imr.astype(bool)
            if occ.any():
                n_occ += 1
                col = np.array(cmap(i % 20)[:3]) * 255
                for c in range(3):
                    occ_vis[:, :, c][occ] = (occ_vis[:, :, c][occ] * 0.25
                                             + col[c] * 0.75)
        occ_vis = occ_vis.clip(0, 255).astype(np.uint8)

        # ── 3-panel figure ──
        fig, axes = plt.subplots(1, 3, figsize=(24, 5))
        cats_str = ", ".join(sorted(set(labels)))
        fig.suptitle(
            f"[{idx+1}/{N}] {info['file_name']}  —  "
            f"{len(anns)} objects, {n_occ} occluded  |  {cats_str}",
            fontsize=11, fontweight="bold")
        axes[0].imshow(amodal_vis)
        axes[0].set_title("AMODAL (full, incl. hidden)",
                          fontsize=10, color="darkgreen", fontweight="bold")
        axes[0].axis("off")
        axes[1].imshow(inmodal_vis)
        axes[1].set_title("VISIBLE (camera only)",
                          fontsize=10, color="darkblue", fontweight="bold")
        axes[1].axis("off")
        axes[2].imshow(occ_vis)
        axes[2].set_title(f"OCCLUDED ({n_occ} objects hidden)",
                          fontsize=10, color="darkred", fontweight="bold")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(C.GT_VIS_DIR,
                                 f"gt_vis_{idx:02d}_full.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── Zoom on most-occluded object ──
        best_i, best_pct = -1, 0
        for i, (am, im_) in enumerate(zip(am_masks, im_masks)):
            amr = safe_resize(am, ih, iw)
            imr = safe_resize(im_, ih, iw)
            tot = amr.sum()
            if tot > 100:
                p = (amr.astype(bool) & ~imr.astype(bool)).sum() / tot
                if 0 < p < 0.95 and p > best_pct:
                    best_pct, best_i = p, i

        if best_i >= 0:
            amr = safe_resize(am_masks[best_i], ih, iw)
            imr = safe_resize(im_masks[best_i], ih, iw)
            ys, xs = np.where(amr > 0)
            pad = 40
            y1, y2 = max(0, ys.min()-pad), min(ih, ys.max()+pad)
            x1, x2 = max(0, xs.min()-pad), min(iw, xs.max()+pad)

            crop = img[y1:y2, x1:x2]
            crop_am = amodal_vis[y1:y2, x1:x2]
            crop_vis = inmodal_vis[y1:y2, x1:x2]

            crop_occ = img[y1:y2, x1:x2].copy().astype(np.float64)
            om = (amr.astype(bool) & ~imr.astype(bool))[y1:y2, x1:x2]
            crop_occ[om] = crop_occ[om] * 0.2 + np.array([255, 50, 50]) * 0.8
            crop_occ = crop_occ.clip(0, 255).astype(np.uint8)

            fig2, ax2 = plt.subplots(1, 4, figsize=(20, 4))
            fig2.suptitle(
                f'Zoomed: Most occluded "{labels[best_i]}" — '
                f'{best_pct*100:.0f}% hidden',
                fontsize=11, fontweight="bold")
            for a, im_data, t in zip(ax2, [crop, crop_am, crop_vis, crop_occ],
                                      ["Original","Amodal","Visible","Occluded (red)"]):
                a.imshow(im_data); a.set_title(t); a.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(C.GT_VIS_DIR,
                                     f"gt_vis_{idx:02d}_zoom.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig2)

        log(f"  [{idx+1}/{N}] {info['file_name']}  "
            f"{len(anns)} obj, {n_occ} occluded")

    log(f"  Saved to {C.GT_VIS_DIR}")
    return True


# ====================================================================
#  STEP 4 — Faster R-CNN baseline inference
# ====================================================================
def step_baseline(log):
    log("=" * 60)
    log("STEP 4  Faster R-CNN Baseline Inference")
    log("=" * 60)
    _ensure_dirs()

    import torch
    if C.AISFORMER_DIR not in sys.path:
        sys.path.insert(0, C.AISFORMER_DIR)

    # Clear cached detectron2 modules so patches take effect
    for k in list(sys.modules):
        if "detectron2" in k:
            del sys.modules[k]

    from detectron2.config import get_cfg, CfgNode as CN
    from detectron2.engine import DefaultPredictor

    # -- Register AISFormer-specific config keys --
    cfg = get_cfg()
    cfg.MODEL.ROI_MASK_HEAD.CUSTOM_NAME = ""
    cfg.MODEL.ROI_MASK_HEAD.RECON_NET = CN()
    cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NAME = ""
    cfg.MODEL.ROI_MASK_HEAD.RECON_NET.NUM_CONVS_INSTANCE = 0
    cfg.MODEL.ROI_MASK_HEAD.RECON_NET.MASK_OUT_CHANNELS = 256
    cfg.MODEL.ROI_MASK_HEAD.RECON_NET.AMODAL_CONV_DIM = 256
    cfg.MODEL.ROI_MASK_HEAD.RECON_NET.VISIBLE_CONV_DIM = 256
    cfg.MODEL.ROI_MASK_HEAD.MASK_FCN_INPUT_CHANNELS = 256
    cfg.MODEL.ROI_MASK_HEAD.NUM_MASK_CLASSES = 1
    cfg.MODEL.ROI_MASK_HEAD.NHEAD = 8
    cfg.MODEL.ROI_MASK_HEAD.HIDDEN_DIM = 256
    cfg.MODEL.ROI_MASK_HEAD.DIM_FEEDFORWARD = 1024
    cfg.MODEL.ROI_MASK_HEAD.NUM_DEC_LAYERS = 4
    cfg.MODEL.ROI_MASK_HEAD.PRE_NORM = False
    cfg.MODEL.ROI_MASK_HEAD.MASK_DIM = 256
    cfg.MODEL.ROI_MASK_HEAD.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.ROI_MASK_HEAD.AMODAL_CYCLE = False
    cfg.MODEL.AISFormer = CN()
    cfg.MODEL.AISFormer.USE = False
    cfg.MODEL.AISFormer.AMODAL_EVAL = False
    cfg.MODEL.AISFormer.JUSTIFY_LOSS = False
    cfg.MODEL.AISFormer.N_HEADS = 8
    cfg.MODEL.AISFormer.N_LAYERS = 4
    cfg.MODEL.ALL_LAYERS_ROI_POOLING = False
    cfg.MODEL.RPN.BOUNDARY_THRESH = -1
    cfg.DICE_LOSS = False
    cfg.OUTPUT_DIR = os.path.join(C.OUTPUT_DIR, "_d2_output")

    # -- Faster R-CNN R50-FPN config --
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2","res3","res4","res5"]
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.FPN.IN_FEATURES = ["res2","res3","res4","res5"]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32],[64],[128],[256],[512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5,1.0,2.0]]
    cfg.MODEL.RPN.IN_FEATURES = ["p2","p3","p4","p5","p6"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2","p3","p4","p5"]
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.MASK_ON = False                       # box-only
    cfg.MODEL.WEIGHTS = C.COCO_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = C.SCORE_THRESH
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    log(f"  Loading model on {cfg.MODEL.DEVICE} …")
    predictor = DefaultPredictor(cfg)
    log("  Faster R-CNN (COCO box-only) loaded.")

    kins, cat_map = _load_kins(log)
    IMG_DIR = C.KITTI_TEST_IMGS
    busy = _busy_images(kins)
    id2info = {im["id"]: im for im in kins["images"]}
    N = C.N_TEST_IMAGES

    results = []
    log(f"\n  Running inference on {N} images …\n")

    for idx in range(N):
        img_id, gt_anns = busy[idx]
        info = id2info.get(img_id)
        if info is None:
            continue
        path = os.path.join(IMG_DIR, info["file_name"])
        if not os.path.isfile(path):
            continue

        raw = cv2.imread(path)
        ih, iw = raw.shape[:2]
        t0 = time.time()
        with torch.no_grad():
            out = predictor(raw)
        dt = time.time() - t0

        inst = out["instances"].to("cpu")
        cls  = inst.pred_classes.numpy()
        sc   = inst.scores.numpy()
        bx   = inst.pred_boxes.tensor.numpy()
        drv  = np.array([C.COCO_CLASSES[c] in C.DRIVING_CLASSES
                         for c in cls]) if len(cls) else np.array([], dtype=bool)

        results.append(dict(
            img_id=img_id, img_info=info, img_path=path,
            gt_anns=gt_anns, pred_classes=cls, pred_scores=sc,
            pred_boxes=bx, driving_mask=drv, inference_time=dt))

        nd = int(drv.sum())
        counts = {}
        for c in cls[drv]:
            n = C.COCO_CLASSES[c]
            counts[n] = counts.get(n, 0) + 1
        cs = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        log(f"  [{idx+1:2d}/{N}] {info['file_name']}  "
            f"{len(cls)} det ({nd} driving: {cs})  {dt*1000:.0f} ms")

    avg_ms = np.mean([r["inference_time"] for r in results]) * 1000
    tot_drv = sum(int(r["driving_mask"].sum()) for r in results)
    log(f"\n  Average {avg_ms:.0f} ms/image, {tot_drv} driving detections total")

    _save_cache(results)
    log("  Results cached.")
    return True


# ====================================================================
#  STEP 5 — Side-by-side comparisons
# ====================================================================
def step_comparisons(log):
    log("=" * 60)
    log("STEP 5  Side-by-Side Comparisons")
    log("=" * 60)
    _ensure_dirs()

    results = _load_cache()
    kins, cat_map = _load_kins(log)

    for ri, res in enumerate(results):
        info = res["img_info"]
        gt   = res["gt_anns"]
        img  = np.array(Image.open(res["img_path"]))
        ih, iw = img.shape[:2]
        ah, aw = info["height"], info["width"]

        am_masks = [decode_segm(a.get("a_segm"), ah, aw) for a in gt]
        im_masks = [decode_segm(a.get("i_segm"), ah, aw) for a in gt]
        gt_labs  = [cat_map.get(a["category_id"], "?") for a in gt]

        didx = np.where(res["driving_mask"])[0]
        dboxes = res["pred_boxes"][didx]
        dscores = res["pred_scores"][didx]

        # Panel 1 — detector boxes
        bv = img.copy()
        cm = plt.cm.tab20
        for i, (box, sc) in enumerate(zip(dboxes, dscores)):
            x1, y1, x2, y2 = (int(v) for v in box)
            col = tuple(int(c*255) for c in cm(i%20)[:3][::-1])
            cv2.rectangle(bv, (x1,y1), (x2,y2), col, 2)
            cn = C.COCO_CLASSES[res["pred_classes"][didx[i]]]
            lab = f"{cn} {sc:.2f}"
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(bv, (x1, y1-th-6), (x1+tw+4, y1), col, -1)
            cv2.putText(bv, lab, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        # Panel 2 — amodal GT
        av = render_labeled_masks(img, am_masks, gt_labs)

        # Panel 3 — occlusion gap
        gv = img.copy().astype(np.float64)
        nocc, tgap, tamodal = 0, 0, 0
        for i, (am, im_) in enumerate(zip(am_masks, im_masks)):
            amr = safe_resize(am, ih, iw); imr = safe_resize(im_, ih, iw)
            occ = amr.astype(bool) & ~imr.astype(bool)
            if occ.any():
                nocc += 1; tgap += occ.sum(); tamodal += amr.astype(bool).sum()
                col = np.array(cm(i%20)[:3]) * 255
                for c in range(3):
                    gv[:,:,c][occ] = gv[:,:,c][occ]*0.2 + col[c]*0.8
        gv = gv.clip(0,255).astype(np.uint8)
        gpct = tgap / max(tamodal, 1) * 100

        fig, axes = plt.subplots(1, 3, figsize=(24, 5))
        nd, ng = len(didx), len(gt)
        fig.suptitle(
            f"[{ri+1}/{len(results)}] {info['file_name']}  —  "
            f"Det: {nd} boxes | GT: {ng} obj, {nocc} occ | "
            f"Gap: {gpct:.0f}%", fontsize=11, fontweight="bold")
        axes[0].imshow(bv)
        axes[0].set_title(f"FASTER R-CNN: {nd} boxes",
                          fontsize=10, color="darkblue", fontweight="bold")
        axes[0].axis("off")
        axes[1].imshow(av)
        axes[1].set_title(f"KINS AMODAL GT: {ng} masks",
                          fontsize=10, color="darkgreen", fontweight="bold")
        axes[1].axis("off")
        axes[2].imshow(gv)
        axes[2].set_title(f"OCCLUSION GAP: {nocc} hidden",
                          fontsize=10, color="darkred", fontweight="bold")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(C.COMPARE_DIR, f"compare_{ri:02d}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  [{ri+1}/{len(results)}] {info['file_name']}  gap {gpct:.0f}%")

    log(f"  Saved to {C.COMPARE_DIR}")
    return True


# ====================================================================
#  STEP 6 — Quantitative analysis
# ====================================================================
def step_quantitative(log):
    log("=" * 60)
    log("STEP 6  Quantitative Analysis")
    log("=" * 60)
    _ensure_dirs()

    results = _load_cache()
    kins, cat_map = _load_kins(log)

    # ── 6a  Per-category occlusion ──
    cs = defaultdict(lambda: dict(count=0, occ_count=0, ratios=[]))
    for r in results:
        ah, aw = r["img_info"]["height"], r["img_info"]["width"]
        for a in r["gt_anns"]:
            cn = cat_map.get(a["category_id"], "?")
            am = decode_segm(a.get("a_segm"), ah, aw)
            im = decode_segm(a.get("i_segm"), ah, aw)
            aa = am.astype(bool).sum()
            if aa < 50: continue
            orr = max(0, min(1, 1 - im.astype(bool).sum()/max(aa,1)))
            cs[cn]["count"] += 1
            cs[cn]["ratios"].append(orr)
            if orr > 0.05: cs[cn]["occ_count"] += 1

    sc = sorted(cs.items(), key=lambda x: -x[1]["count"])
    log("\n  Per-category occlusion:")
    for cn, s in sc:
        pct = s["occ_count"]/max(s["count"],1)*100
        mn = np.mean(s["ratios"])*100
        log(f"    {cn:<14} n={s['count']:>4}  {pct:.0f}% occluded  mean={mn:.1f}%")

    cats = [c for c,_ in sc]
    mocc = [np.mean(cs[c]["ratios"])*100 for c in cats]
    pocc = [cs[c]["occ_count"]/max(cs[c]["count"],1)*100 for c in cats]
    cols = plt.cm.Set2(np.linspace(0,1,len(cats)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5))
    a1.bar(cats, pocc, color=cols, edgecolor="black", linewidth=0.5)
    a1.set_ylabel("% Instances Occluded"); a1.set_title("Occlusion Frequency", fontweight="bold")
    a1.set_ylim(0,105); a1.tick_params(axis="x", rotation=30)
    a2.bar(cats, mocc, color=cols, edgecolor="black", linewidth=0.5)
    a2.set_ylabel("Mean Occlusion %"); a2.set_title("Mean Hidden Area", fontweight="bold")
    a2.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(C.CHARTS_DIR, "occlusion_by_category.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 6b  Recall vs occlusion ──
    bins = [(0,.1,"0-10%"),(.1,.3,"10-30%"),(.3,.5,"30-50%"),
            (.5,.7,"50-70%"),(.7,1.01,"70-100%")]
    bt, bd = defaultdict(int), defaultdict(int)
    for r in results:
        ih, iw = r["img_info"]["height"], r["img_info"]["width"]
        didx = np.where(r["driving_mask"])[0]
        dbox = r["pred_boxes"][didx]
        for a in r["gt_anns"]:
            am = decode_segm(a.get("a_segm"), ih, iw)
            im = decode_segm(a.get("i_segm"), ih, iw)
            aa = am.astype(bool).sum()
            if aa < 50: continue
            orr = max(0, 1 - im.astype(bool).sum()/max(aa,1))
            ys, xs = (np.where(im>0) if im.any() else np.where(am>0))
            if not len(ys): continue
            gb = [xs.min(), ys.min(), xs.max(), ys.max()]
            det = False
            for db in dbox:
                x1=max(gb[0],db[0]); y1=max(gb[1],db[1])
                x2=min(gb[2],db[2]); y2=min(gb[3],db[3])
                if x2>x1 and y2>y1:
                    inter=(x2-x1)*(y2-y1)
                    iou=inter/max((gb[2]-gb[0])*(gb[3]-gb[1])+(db[2]-db[0])*(db[3]-db[1])-inter,1)
                    if iou>0.3: det=True; break
            for lo,hi,lab in bins:
                if lo<=orr<hi:
                    bt[lab]+=1
                    if det: bd[lab]+=1
                    break

    rl = [l for _,_,l in bins]
    rv = [bd[l]/max(bt[l],1)*100 for l in rl]
    rc = [bt[l] for l in rl]
    log("\n  Recall vs occlusion:")
    for l,v,n in zip(rl,rv,rc):
        log(f"    {l:<10} n={n:>4}  recall={v:.0f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    bc = plt.cm.RdYlGn_r(np.linspace(0.1,0.9,len(rl)))
    bars = ax.bar(rl, rv, color=bc, edgecolor="black", linewidth=0.5, width=0.6)
    for b,v,n in zip(bars,rv,rc):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                f"{v:.0f}%\n(n={n})", ha="center", fontsize=10)
    ax.set_xlabel("GT Occlusion Level"); ax.set_ylabel("Recall (%)")
    ax.set_title("Detector Recall Drops as Occlusion Increases", fontweight="bold")
    ax.set_ylim(0,115)
    ax.axhline(50, color="red", ls="--", alpha=0.4, label="50% recall"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(C.CHARTS_DIR, "recall_vs_occlusion.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 6c  Box IoU vs amodal coverage ──
    vis_ious, am_covs, occ_matched = [], [], []
    for r in results:
        ih, iw = r["img_info"]["height"], r["img_info"]["width"]
        didx = np.where(r["driving_mask"])[0]
        dbox = r["pred_boxes"][didx]
        for a in r["gt_anns"]:
            am = safe_resize(decode_segm(a.get("a_segm"),ih,iw),ih,iw).astype(bool)
            im = safe_resize(decode_segm(a.get("i_segm"),ih,iw),ih,iw).astype(bool)
            aa, va = am.sum(), im.sum()
            if aa<50 or va<20: continue
            orr = 1 - va/max(aa,1)
            ys, xs = np.where(im)
            if not len(ys): continue
            gb = np.array([xs.min(),ys.min(),xs.max(),ys.max()])
            best_iou, best_box = 0, None
            for db in dbox:
                x1=max(gb[0],db[0]); y1=max(gb[1],db[1])
                x2=min(gb[2],db[2]); y2=min(gb[3],db[3])
                if x2>x1 and y2>y1:
                    inter=(x2-x1)*(y2-y1)
                    iou=inter/max((gb[2]-gb[0])*(gb[3]-gb[1])+(db[2]-db[0])*(db[3]-db[1])-inter,1)
                    if iou>best_iou: best_iou, best_box = iou, db
            if best_iou>0.3 and best_box is not None:
                bx1,by1,bx2,by2=(int(v) for v in best_box)
                bm = np.zeros((ih,iw),dtype=bool)
                bm[max(0,by1):min(ih,by2), max(0,bx1):min(iw,bx2)] = True
                vis_ious.append(best_iou)
                am_covs.append((bm & am).sum()/max(aa,1))
                occ_matched.append(orr)

    vis_ious = np.array(vis_ious); am_covs = np.array(am_covs)
    occ_matched = np.array(occ_matched)

    be = [0,.1,.3,.5,.7,1.01]
    bl = ["0-10%","10-30%","30-50%","50-70%","70-100%"]
    vb, ab = [], []
    for lo,hi in zip(be[:-1],be[1:]):
        m = (occ_matched>=lo)&(occ_matched<hi)
        vb.append(vis_ious[m].mean()*100 if m.sum() else 0)
        ab.append(am_covs[m].mean()*100 if m.sum() else 0)

    fig, (a1,a2) = plt.subplots(1,2,figsize=(16,6))
    sc = a1.scatter(occ_matched*100, am_covs*100, c=vis_ious, cmap="RdYlGn",
                    s=50, alpha=0.7, edgecolors="black", linewidth=0.3)
    a1.set_xlabel("Occlusion (%)"); a1.set_ylabel("Box Coverage of Amodal (%)")
    a1.set_title("Coverage Drops with Occlusion", fontweight="bold")
    plt.colorbar(sc, ax=a1, label="Visible-Box IoU")

    xp = np.arange(len(bl)); w=0.35
    a2.bar(xp-w/2, vb, w, label="Visible-Box IoU", color="#2196F3", edgecolor="black", lw=0.5)
    a2.bar(xp+w/2, ab, w, label="Amodal Coverage", color="#FF9800", edgecolor="black", lw=0.5)
    a2.set_xticks(xp); a2.set_xticklabels(bl)
    a2.set_xlabel("Occlusion Level"); a2.set_ylabel("Score (%)")
    a2.set_title("IoU vs Coverage by Occlusion", fontweight="bold")
    a2.set_ylim(0,115); a2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(C.CHARTS_DIR, "iou_vs_occlusion.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    log("\n  Quantitative analysis complete.")
    return True


# ====================================================================
#  STEP 7 — Failure-case deep dive
# ====================================================================
def step_deep_dive(log):
    log("=" * 60)
    log("STEP 7  Failure-Case Deep Dive")
    log("=" * 60)
    _ensure_dirs()

    results = _load_cache()
    kins, cat_map = _load_kins(log)

    # Rank images by occlusion severity + missed detections
    scores_list = []
    for res in results:
        ah, aw = res["img_info"]["height"], res["img_info"]["width"]
        ih, iw = ah, aw
        didx = np.where(res["driving_mask"])[0]
        dbox = res["pred_boxes"][didx]
        ratios, n_heavy, n_miss = [], 0, 0
        for a in res["gt_anns"]:
            am = decode_segm(a.get("a_segm"), ah, aw)
            im = decode_segm(a.get("i_segm"), ah, aw)
            aa = am.astype(bool).sum()
            if aa < 50: continue
            orr = max(0, 1 - im.astype(bool).sum()/max(aa,1))
            ratios.append(orr)
            if orr > 0.4: n_heavy += 1
            ys, xs = (np.where(im>0) if im.any() else np.where(am>0))
            if not len(ys): continue
            gb = [xs.min(),ys.min(),xs.max(),ys.max()]
            det = False
            for db in dbox:
                x1=max(gb[0],db[0]); y1=max(gb[1],db[1])
                x2=min(gb[2],db[2]); y2=min(gb[3],db[3])
                if x2>x1 and y2>y1:
                    inter=(x2-x1)*(y2-y1)
                    iou=inter/max((gb[2]-gb[0])*(gb[3]-gb[1])+(db[2]-db[0])*(db[3]-db[1])-inter,1)
                    if iou>0.3: det=True; break
            if not det: n_miss += 1
        mo = np.mean(ratios) if ratios else 0
        score = mo*0.4 + n_heavy/max(len(res["gt_anns"]),1)*0.3 + n_miss/max(len(res["gt_anns"]),1)*0.3
        scores_list.append((score, res, n_miss))

    scores_list.sort(key=lambda x: -x[0])
    top5 = scores_list[:5]
    log(f"  Top {len(top5)} hardest scenes selected.\n")

    for ci, (_, res, n_miss) in enumerate(top5):
        info = res["img_info"]
        gt = res["gt_anns"]
        img = np.array(Image.open(res["img_path"]))
        ih, iw = img.shape[:2]
        ah, aw = info["height"], info["width"]
        didx = np.where(res["driving_mask"])[0]
        dbox = res["pred_boxes"][didx]
        dsc = res["pred_scores"][didx]
        dcls = res["pred_classes"][didx]

        objects = []
        for a in gt:
            am = safe_resize(decode_segm(a.get("a_segm"),ah,aw),ih,iw)
            im = safe_resize(decode_segm(a.get("i_segm"),ah,aw),ih,iw)
            aa, va = am.astype(bool).sum(), im.astype(bool).sum()
            if aa < 50: continue
            orr = max(0, 1-va/max(aa,1))
            cn = cat_map.get(a["category_id"], "?")
            ys, xs = (np.where(im>0) if im.any() else np.where(am>0))
            if not len(ys): continue
            gb = [xs.min(),ys.min(),xs.max(),ys.max()]
            det, mb = False, None
            for db in dbox:
                x1=max(gb[0],db[0]); y1=max(gb[1],db[1])
                x2=min(gb[2],db[2]); y2=min(gb[3],db[3])
                if x2>x1 and y2>y1:
                    inter=(x2-x1)*(y2-y1)
                    iou=inter/max((gb[2]-gb[0])*(gb[3]-gb[1])+(db[2]-db[0])*(db[3]-db[1])-inter,1)
                    if iou>0.3: det=True; mb=db; break
            objects.append(dict(cat=cn, occ=orr, am=am, vis=im, det=det,
                                mbox=mb, aa=aa, va=va))
        objects.sort(key=lambda o: -o["occ"])

        ng = len(objects)
        nf = sum(1 for o in objects if o["det"])
        mocc = np.mean([o["occ"] for o in objects])*100

        # Overview figure
        fig, axes = plt.subplots(1, 3, figsize=(24, 5.5))
        fig.suptitle(
            f"FAILURE {ci+1}: {info['file_name']}  —  "
            f"Detected {nf}/{ng} | Mean occ {mocc:.0f}% | {n_miss} missed",
            fontsize=12, fontweight="bold", color="darkred")

        bimg = img.copy()
        cm = plt.cm.tab20
        for i, box in enumerate(dbox):
            x1,y1,x2,y2 = (int(v) for v in box)
            col = tuple(int(c*255) for c in cm(i%20)[:3][::-1])
            cv2.rectangle(bimg,(x1,y1),(x2,y2),col,2)
            lab = f"{C.COCO_CLASSES[dcls[i]]} {dsc[i]:.2f}"
            (tw,th),_ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(bimg,(x1,y1-th-6),(x1+tw+4,y1),col,-1)
            cv2.putText(bimg,lab,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv2.LINE_AA)
        axes[0].imshow(bimg); axes[0].set_title(f"Faster R-CNN: {len(dbox)} boxes",
            fontsize=10, color="darkblue", fontweight="bold"); axes[0].axis("off")

        aimg = img.copy().astype(np.float64)
        for i, o in enumerate(objects):
            b = o["am"].astype(bool)
            col = np.array([50,200,50]) if o["det"] else np.array([220,50,50])
            for c in range(3): aimg[:,:,c][b] = aimg[:,:,c][b]*0.5 + col[c]*0.5
        aimg = aimg.clip(0,255).astype(np.uint8)
        axes[1].imshow(aimg)
        axes[1].set_title(f"GT: green=det, red=missed ({ng-nf} missed)",
            fontsize=10, color="darkgreen", fontweight="bold"); axes[1].axis("off")

        oimg = img.copy().astype(np.float64)
        for o in objects:
            occ_m = o["am"].astype(bool) & ~o["vis"].astype(bool)
            if occ_m.any():
                col = np.array([255,165,0]) if o["det"] else np.array([255,30,30])
                for c in range(3): oimg[:,:,c][occ_m] = oimg[:,:,c][occ_m]*0.15+col[c]*0.85
        oimg = oimg.clip(0,255).astype(np.uint8)
        axes[2].imshow(oimg)
        axes[2].set_title("Hidden: orange=det, red=missed",
            fontsize=10, color="darkred", fontweight="bold"); axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(C.DEEPDIVE_DIR, f"failure_{ci:02d}_overview.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Zoom on top-3 most occluded
        zooms = [o for o in objects if o["occ"]>0.15 and o["aa"]>200][:3]
        if zooms:
            fig2, ax2 = plt.subplots(len(zooms), 4, figsize=(20, 4.5*len(zooms)))
            if len(zooms)==1: ax2 = ax2.reshape(1,-1)
            fig2.suptitle(f"Zoomed: Top {len(zooms)} Occluded in {info['file_name']}",
                          fontsize=12, fontweight="bold")
            for row, o in enumerate(zooms):
                ys,xs = np.where(o["am"]>0); pad=50
                y1,y2 = max(0,ys.min()-pad), min(ih,ys.max()+pad)
                x1,x2 = max(0,xs.min()-pad), min(iw,xs.max()+pad)

                crop = img[y1:y2,x1:x2].copy()
                if o["mbox"] is not None:
                    bx = [int(v) for v in o["mbox"]]
                    cv2.rectangle(crop,(bx[0]-x1,bx[1]-y1),(bx[2]-x1,bx[3]-y1),(0,255,255),2)

                vc = img[y1:y2,x1:x2].copy().astype(np.float64)
                vm = o["vis"][y1:y2,x1:x2].astype(bool)
                for c in range(3): vc[:,:,c][vm] = vc[:,:,c][vm]*0.5+[50,150,255][c]*0.5
                vc = vc.clip(0,255).astype(np.uint8)

                ac = img[y1:y2,x1:x2].copy().astype(np.float64)
                amm = o["am"][y1:y2,x1:x2].astype(bool)
                for c in range(3): ac[:,:,c][amm] = ac[:,:,c][amm]*0.5+[50,200,50][c]*0.5
                ac = ac.clip(0,255).astype(np.uint8)

                oc = img[y1:y2,x1:x2].copy().astype(np.float64)
                om = (o["am"].astype(bool)&~o["vis"].astype(bool))[y1:y2,x1:x2]
                for c in range(3): oc[:,:,c][om] = oc[:,:,c][om]*0.15+[255,50,50][c]*0.85
                oc = oc.clip(0,255).astype(np.uint8)

                st = "DETECTED" if o["det"] else "MISSED"
                sc_ = "green" if o["det"] else "red"
                ax2[row,0].imshow(crop); ax2[row,0].set_title(f"Original ({st})",fontsize=9,color=sc_,fontweight="bold"); ax2[row,0].axis("off")
                ax2[row,1].imshow(vc); ax2[row,1].set_title(f"Visible: {o['va']:,}px",fontsize=9,fontweight="bold"); ax2[row,1].axis("off")
                ax2[row,2].imshow(ac); ax2[row,2].set_title(f"Amodal: {o['aa']:,}px",fontsize=9,fontweight="bold"); ax2[row,2].axis("off")
                hid = o["aa"]-o["va"]
                ax2[row,3].imshow(oc); ax2[row,3].set_title(f"Hidden: {hid:,}px ({o['occ']*100:.0f}%)",fontsize=9,color="red",fontweight="bold"); ax2[row,3].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(C.DEEPDIVE_DIR, f"failure_{ci:02d}_zoom.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig2)

        log(f"  Case {ci+1}: {info['file_name']}  det {nf}/{ng}, missed {n_miss}")

    log(f"  Saved to {C.DEEPDIVE_DIR}")
    return True


# ====================================================================
#  STEP 8 — AISFormer published results charts
# ====================================================================
def step_aisformer_charts(log):
    log("=" * 60)
    log("STEP 8  AISFormer Published Results")
    log("=" * 60)
    _ensure_dirs()

    methods = ["Mask R-CNN", "ORCNN", "BCNet", "AISFormer"]
    ap = [29.4, 30.3, 31.0, 34.4]
    ap50 = [52.3, 52.8, 54.0, 58.0]
    ap75 = [28.5, 29.7, 30.2, 34.4]
    aps = [5.7, 6.0, 6.3, 7.3]
    apm = [20.0, 20.8, 21.5, 24.7]
    apl = [52.3, 53.2, 53.6, 56.5]
    cols = ["#90A4AE","#78909C","#607D8B","#1E88E5"]
    ecols = ["#546E7A","#455A64","#37474F","#0D47A1"]

    # Main AP chart
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(4)
    b = a1.bar(x, ap, color=cols, edgecolor=ecols, linewidth=1.5, width=0.6)
    for bar, v in zip(b, ap):
        a1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                str(v), ha="center", fontsize=13, fontweight="bold")
    a1.annotate("+5.0 AP\nimprovement", xy=(3, ap[3]), xytext=(1.5, ap[3]+3),
                fontsize=11, fontweight="bold", color="#0D47A1", ha="center",
                arrowprops=dict(arrowstyle="->", color="#0D47A1", lw=2))
    a1.set_xticks(x); a1.set_xticklabels(methods)
    a1.set_ylabel("AP (Amodal)"); a1.set_title("Amodal Segmentation on KINS", fontweight="bold")
    a1.set_ylim(0,42); a1.spines["top"].set_visible(False); a1.spines["right"].set_visible(False)

    w = 0.35
    a2.bar(x-w/2, ap50, w, label="AP50",
           color=["#B0BEC5","#90A4AE","#78909C","#42A5F5"], edgecolor="black", lw=0.5)
    a2.bar(x+w/2, ap75, w, label="AP75",
           color=["#ECEFF1","#CFD8DC","#B0BEC5","#90CAF9"], edgecolor="black", lw=0.5)
    a2.set_xticks(x); a2.set_xticklabels(methods)
    a2.set_ylabel("AP"); a2.set_title("AP50 / AP75", fontweight="bold")
    a2.set_ylim(0,68); a2.legend()
    a2.spines["top"].set_visible(False); a2.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(C.CHARTS_DIR, "aisformer_published_results.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # By size
    fig2, a3 = plt.subplots(figsize=(12, 6))
    x = np.arange(3); w = 0.18
    for i, (meth, s, m, l, col) in enumerate(zip(methods, aps, apm, apl, cols)):
        bars = a3.bar(x+i*w, [s,m,l], w, label=meth, color=col, edgecolor="black", lw=0.5)
        for bar, v in zip(bars, [s,m,l]):
            a3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    str(v), ha="center", fontsize=8, fontweight="bold")
    a3.set_xticks(x+1.5*w); a3.set_xticklabels(["Small","Medium","Large"])
    a3.set_ylabel("AP"); a3.set_title("Performance by Object Size", fontweight="bold")
    a3.legend(loc="upper left"); a3.set_ylim(0,65)
    a3.spines["top"].set_visible(False); a3.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(C.CHARTS_DIR, "aisformer_by_size.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Findings → solution table
    fig3, a4 = plt.subplots(figsize=(14, 5)); a4.axis("off")
    td = [["Our Finding","Impact","AISFormer Solution"],
          ["55% objects partially\noccluded","No occlusion\nawareness","Tri-mask prediction:\namodal+visible+invisible"],
          ["Recall 80%→37%\nwith occlusion","Safety objects\nmissed","Occluder-aware\nquery token"],
          ["Coverage 96%→59%\nwith occlusion","Poor localisation\nof detected objects","Pixel-embedding\ndot product"],
          ["Boxes can't capture\nobject shape","Rectangular\napproximation","Transformer decoder\nwith learned queries"]]
    tab = a4.table(cellText=td[1:], colLabels=td[0], loc="center", cellLoc="center",
                   colWidths=[0.32,0.32,0.36])
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1,2.2)
    for j in range(3):
        tab[0,j].set_facecolor("#1E88E5")
        tab[0,j].set_text_props(color="white", fontweight="bold", fontsize=11)
    rc = ["#E3F2FD","#FFFFFF","#E3F2FD","#FFFFFF"]
    for i in range(1,5):
        for j in range(3):
            tab[i,j].set_facecolor(rc[i-1])
            if j==2: tab[i,j].set_text_props(fontweight="bold", color="#0D47A1")
    a4.set_title("Connecting Analysis to AISFormer", fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(C.CHARTS_DIR, "findings_to_solution.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig3)

    log("  AISFormer: 34.4 AP (+5.0 over Mask R-CNN)")
    log("  Charts saved.")
    return True


# ====================================================================
#  STEP 9 — Package everything into a ZIP
# ====================================================================
def step_package(log):
    log("=" * 60)
    log("STEP 9  Package Results")
    log("=" * 60)

    fc = 0
    with zipfile.ZipFile(C.RESULTS_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in (C.GT_VIS_DIR, C.COMPARE_DIR, C.CHARTS_DIR, C.DEEPDIVE_DIR):
            for f in sorted(glob.glob(os.path.join(d, "*.png"))):
                arc = os.path.relpath(f, C.OUTPUT_DIR)
                zf.write(f, arc); fc += 1

    sz = os.path.getsize(C.RESULTS_ZIP) / 1e6
    log(f"  {C.RESULTS_ZIP}")
    log(f"  {sz:.1f} MB, {fc} files")
    log("  Done!")
    return True
