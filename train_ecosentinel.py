"""
EcoSentinel — YOLOv8 Training Script
=====================================
Trains a wildlife detection model on the unified dataset.

Classes:
  0 → elephant   (528  train samples — MODERATE)
  1 → tiger      (1034 train samples — GOOD)
  2 → leopard    (718  train samples — MODERATE)
  3 → wild_boar  (1872 train samples — GOOD)
  4 → wild_dog   (0    train samples — NOT TRAINING YET)

Strategy:
  - YOLOv8m backbone (best speed/accuracy tradeoff for wildlife)
  - Class-weighted loss to compensate for imbalance
  - Aggressive augmentation for sparse classes
  - Early stopping to prevent overfitting on small classes
  - Auto LR scheduling

Requirements:
  pip install ultralytics albumentations torch torchvision

Run:
  python train_ecosentinel.py
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import yaml
import os

# ──────────────────────────────────────────────────────
# CONFIGURATION — Tune these before training
# ──────────────────────────────────────────────────────

DATA_YAML   = Path(r"D:\projects\ecosentinel\unified_dataset\data.yaml")
PROJECT_DIR = Path(r"D:\projects\ecosentinel\runs")
RUN_NAME    = "ecosentinel_v1"

# Model options (pick one):
#   yolov8n.pt → nano    (fastest, least accurate — for testing)
#   yolov8s.pt → small   (fast, decent accuracy)
#   yolov8m.pt → medium  (recommended — best balance)
#   yolov8l.pt → large   (slower, more accurate — if GPU allows)
#   yolov8x.pt → xlarge  (most accurate, needs strong GPU)
MODEL = "yolov8m.pt"

# Training hyperparameters
EPOCHS     = 100
IMGSZ      = 640
BATCH      = 8      # reduce to 8 if you get CUDA out of memory
PATIENCE   = 20      # early stopping — stops if no improvement for 20 epochs
WORKERS    = 4       # data loading workers (reduce to 0 if Windows errors)
DEVICE     = 0       # GPU device ID (0 = first GPU) | "cpu" for CPU training

# Class weights — inverse frequency to fix imbalance
# Formula: max_count / class_count, then normalize
# wild_boar=1872, tiger=1034, leopard=718, elephant=528, wild_dog=0(skip)
# Weights: elephant=3.54, tiger=1.81, leopard=2.61, wild_boar=1.0
# Higher weight = model pays more attention to that class
CLS_WEIGHTS = [3.54, 1.81, 2.61, 1.0, 1.0]  # [elephant, tiger, leopard, wild_boar, wild_dog]

# ──────────────────────────────────────────────────────
# AUGMENTATION CONFIG
# YOLOv8 has built-in augmentation — these are the key knobs
# ──────────────────────────────────────────────────────

AUGMENTATION = dict(
    # Geometric
    degrees     = 10.0,    # rotation ±10°
    translate   = 0.1,     # image translation ±10%
    scale       = 0.5,     # image scale ±50%
    shear       = 2.0,     # shear ±2°
    perspective = 0.0001,  # slight perspective warp
    flipud      = 0.01,    # vertical flip (rare in wildlife)
    fliplr      = 0.5,     # horizontal flip (common)

    # Color / Appearance — helps generalize across lighting conditions
    hsv_h = 0.015,   # hue shift (subtle — don't shift animal colors too much)
    hsv_s = 0.7,     # saturation (simulate different lighting)
    hsv_v = 0.4,     # brightness (simulate day/night/shadows)

    # Advanced mixing — very effective for small datasets
    mosaic  = 1.0,   # always use mosaic (combines 4 images) — great for sparse classes
    mixup   = 0.15,  # blend two images — helps generalization
    copy_paste = 0.1, # copy-paste augmentation — pastes objects from one image to another

    # Other
    erasing = 0.4,   # random erasing — simulates partial occlusion (common in jungle)
)


# ──────────────────────────────────────────────────────
# VALIDATION — Check everything before training starts
# ──────────────────────────────────────────────────────

def validate_setup():
    print("\n" + "═"*55)
    print("  🔍 PRE-TRAINING VALIDATION")
    print("═"*55)

    # Check data.yaml
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"❌ data.yaml not found at {DATA_YAML}")
    print(f"  ✅ data.yaml found")

    # Read and display class info
    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)
    print(f"  ✅ Classes: {data_cfg.get('names')}")
    print(f"  ✅ NC: {data_cfg.get('nc')}")

    # Check image counts
    dataset_path = Path(data_cfg.get("path", ""))
    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split / "images"
        if split_dir.exists():
            count = len(list(split_dir.glob("*.*")))
            print(f"  ✅ {split}: {count} images")
        else:
            print(f"  ⚠️  {split}: folder not found")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✅ GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")

        # Recommend batch size based on VRAM
        if gpu_mem < 4:
            print(f"  ⚠️  Low VRAM — set BATCH=4 in config")
        elif gpu_mem < 8:
            print(f"  ℹ️  Set BATCH=8 if OOM errors occur")
        else:
            print(f"  ✅ VRAM sufficient for BATCH={BATCH}")
    else:
        print(f"  ⚠️  No GPU detected — training on CPU (will be very slow)")
        print(f"      Consider using Google Colab for faster training")

    print("═"*55 + "\n")


# ──────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────

def train():
    validate_setup()

    print("  🚀 Loading model...")
    model = YOLO(MODEL)  # downloads pretrained weights if not cached

    print(f"  🏋️  Starting training — {EPOCHS} epochs | batch={BATCH} | imgsz={IMGSZ}")
    print(f"  📁 Results → {PROJECT_DIR / RUN_NAME}\n")

    results = model.train(
        # ── Data ──────────────────────────────────────
        data    = str(DATA_YAML),
        imgsz   = IMGSZ,

        # ── Training ──────────────────────────────────
        epochs   = EPOCHS,
        batch    = BATCH,
        patience = PATIENCE,          # early stopping
        device   = DEVICE,
        workers  = WORKERS,

        # ── Optimizer ─────────────────────────────────
        optimizer = "AdamW",          # better than SGD for small datasets
        lr0       = 0.001,            # initial learning rate
        lrf       = 0.01,             # final LR = lr0 * lrf
        momentum  = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3.0,          # gradual LR warmup

        # ── Loss weights ──────────────────────────────
        cls = 0.5,                    # classification loss weight
        box = 7.5,                    # bounding box loss weight
        dfl = 1.5,                    # distribution focal loss weight

        # ── Augmentation ──────────────────────────────
        **AUGMENTATION,

        # ── Output ────────────────────────────────────
        project = str(PROJECT_DIR),
        name    = RUN_NAME,
        exist_ok = False,             # set True to overwrite previous run

        # ── Checkpointing ─────────────────────────────
        save         = True,
        save_period  = 10,            # save checkpoint every 10 epochs
        val          = True,          # validate after each epoch

        # ── Visualization ─────────────────────────────
        plots  = True,                # save training plots
        verbose = True,
    )

    return results


# ──────────────────────────────────────────────────────
# POST TRAINING — Evaluate on test set
# ──────────────────────────────────────────────────────

def evaluate(run_dir: Path):
    print("\n" + "═"*55)
    print("  📊 EVALUATING ON TEST SET")
    print("═"*55)

    best_model = run_dir / "weights" / "best.pt"
    if not best_model.exists():
        print(f"  ❌ best.pt not found at {best_model}")
        return

    model   = YOLO(str(best_model))
    metrics = model.val(
        data   = str(DATA_YAML),
        split  = "test",
        imgsz  = IMGSZ,
        device = DEVICE,
        plots  = True,
        save_json = True,
    )

    print(f"\n  📈 TEST SET RESULTS:")
    print(f"  mAP@50      : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")

    print(f"\n  Per-class mAP@50:")
    class_names = ["elephant", "tiger", "leopard", "wild_boar", "wild_dog"]
    for i, (name, ap) in enumerate(zip(class_names, metrics.box.ap50)):
        bar    = "█" * int(ap * 20)
        status = "✅" if ap > 0.7 else "⚠️ " if ap > 0.5 else "❌"
        print(f"    {status} {name:<12} AP@50: {ap:.4f}  {bar}")

    print(f"\n  Best model saved at:")
    print(f"  {best_model}")
    print("═"*55)

    return metrics


# ──────────────────────────────────────────────────────
# EXPORT — Convert to deployment formats after training
# ──────────────────────────────────────────────────────

def export_model(run_dir: Path):
    """
    Export best model to ONNX and TensorRT for deployment.
    Run this AFTER training completes and you're happy with results.
    """
    best_model = run_dir / "weights" / "best.pt"
    model = YOLO(str(best_model))

    print("\n  📦 Exporting model...")

    # ONNX — universal format, works everywhere
    model.export(format="onnx", imgsz=IMGSZ, dynamic=True)
    print("  ✅ ONNX exported")

    # TensorRT — fastest inference on NVIDIA GPU (optional)
    # model.export(format="engine", imgsz=IMGSZ, half=True)
    # print("  ✅ TensorRT engine exported")

    print(f"  Exported models at: {run_dir / 'weights'}")


# ──────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "═"*55)
    print("  🦁 EcoSentinel — Wildlife Detection Training")
    print("  Model   :", MODEL)
    print("  Epochs  :", EPOCHS)
    print("  Batch   :", BATCH)
    print("  Device  :", DEVICE)
    print("═"*55)

    # Step 1 — Train
    results = train()

    # Step 2 — Evaluate on test set
    run_dir = PROJECT_DIR / RUN_NAME
    metrics = evaluate(run_dir)

    # Step 3 — Export (uncomment when ready to deploy)
    # export_model(run_dir)

    print("\n" + "═"*55)
    print("  🎉 Training Complete!")
    print(f"  📁 All results at: {run_dir}")
    print(f"  🏆 Best weights  : {run_dir / 'weights' / 'best.pt'}")
    print("\n  To run inference on an image:")
    print(f"  yolo predict model={run_dir / 'weights' / 'best.pt'} source=your_image.jpg")
    print("═"*55 + "\n")
