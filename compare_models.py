"""
EcoSentinel — Model Comparison Benchmark
==========================================
Evaluates multiple YOLO variants on the SAME test set
to produce a fair comparison table for the research paper.

All models are trained on the SAME unified_dataset with
identical hyperparameters — only architecture changes.

Run:
  python compare_models.py

Output:
  comparison_results.csv  — raw numbers
  comparison_table.txt    — formatted table for paper
"""

from ultralytics import YOLO
from pathlib import Path
import csv
import time
import torch

# ──────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────

DATA_YAML   = Path(r"D:\projects\ecosentinel\unified_dataset\data.yaml")
OUTPUT_DIR  = Path(r"D:\projects\ecosentinel\comparison")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = 0 if torch.cuda.is_available() else "cpu"
IMGSZ  = 640

# ──────────────────────────────────────────
# MODELS TO COMPARE
# Each gets trained fresh on your dataset
# then evaluated on your test set
# ──────────────────────────────────────────

MODELS = [
    #{
    #   "name":    "YOLOv5s",
    #   "weights": "yolov5su.pt",   # ultralytics reimplementation
    #   "type":    "yolo",
    #   "params":  "7.2M",
    #   "note":    "Baseline — predecessor architecture",
    #},
  #  {
   #     "name":    "YOLOv8n",
    #    "weights": "yolov8n.pt",    # nano — smallest variant
    #    "type":    "yolo",
     #   "params":  "3.2M",
      #  "note":    "Smallest YOLOv8 — speed benchmark",
    #},
    {
        "name":    "YOLOv8s (Ours)",
        "weights": r"D:\projects\ecosentinel\runs\ecosentinel_v1\weights\best.pt",
        "type":    "pretrained",    # already trained — just evaluate
        "params":  "11.2M",
        "note":    "EcoSentinel — proposed method",
    },
   # {
    #    "name":    "YOLOv8m",
     #   "weights": "yolov8m.pt",    # medium — larger comparison
      #  "type":    "yolo",
       # "params":  "25.9M",
        #"note":    "Larger variant comparison",
    #},
   # {
    #    "name":    "YOLOv9s",
     #   "weights": "yolov9s.pt",
      #  "type":    "yolo",
       # "params":  "7.1M",
        #"note":    "Latest YOLO generation",
    #},
    {
        "name":    "YOLOv10s",
        "weights": "yolov10s.pt",
        "type":    "yolo",
        "params":  "8.0M",
        "note":    "NMS-free detection head",
    },
]

# Training config for models that need training
# Use fewer epochs for comparison — fair quick benchmark
TRAIN_EPOCHS  = 20    # half of your main training
TRAIN_BATCH   = 16
TRAIN_PATIENCE = 15


# ──────────────────────────────────────────
# TRAIN A MODEL FROM SCRATCH
# ──────────────────────────────────────────

def train_model(model_cfg: dict) -> Path:
    """Train a model on the unified dataset and return weights path."""
    name = model_cfg["name"].replace(" ", "_").replace("(", "").replace(")", "")
    run_name = f"compare_{name}"

    print(f"\n{'='*55}")
    print(f"  🏋️  Training: {model_cfg['name']}")
    print(f"  Weights: {model_cfg['weights']}")
    print(f"{'='*55}")

    model = YOLO(model_cfg["weights"])
    model.train(
        data      = str(DATA_YAML),
        epochs    = TRAIN_EPOCHS,
        batch     = TRAIN_BATCH,
        imgsz     = IMGSZ,
        device    = DEVICE,
        patience  = TRAIN_PATIENCE,
        optimizer = "AdamW",
        lr0       = 0.001,
        warmup_epochs = 3,
        workers   = 0,
        cache     = True,
        mosaic    = 1.0,
        mixup     = 0.1,
        project   = str(OUTPUT_DIR / "runs"),
        name      = run_name,
        exist_ok  = True,
        verbose   = False,
    )

    best_weights = OUTPUT_DIR / "runs" / run_name / "weights" / "best.pt"
    return best_weights


# ──────────────────────────────────────────
# EVALUATE A MODEL ON TEST SET
# ──────────────────────────────────────────

def evaluate_model(weights_path: str | Path, model_name: str) -> dict:
    """Evaluate a model on the test set and return metrics dict."""
    print(f"\n  📊 Evaluating: {model_name}")

    model = YOLO(str(weights_path))

    # Measure inference speed
    start = time.time()
    metrics = model.val(
        data    = str(DATA_YAML),
        split   = "test",
        imgsz   = IMGSZ,
        device  = DEVICE,
        verbose = False,
        plots   = False,
    )
    elapsed = time.time() - start

    # Count test images for FPS calculation
    test_img_dir = Path(str(DATA_YAML).replace("data.yaml", "")) / "test" / "images"
    n_images = len(list(test_img_dir.glob("*.*"))) if test_img_dir.exists() else 474

    fps = n_images / elapsed if elapsed > 0 else 0

    # Count model parameters
    param_count = sum(p.numel() for p in model.model.parameters()) / 1e6

    result = {
        "model":     model_name,
        "map50":     round(metrics.box.map50, 4),
        "map5095":   round(metrics.box.map, 4),
        "precision": round(metrics.box.mp, 4),
        "recall":    round(metrics.box.mr, 4),
        "params_m":  round(param_count, 1),
        "fps":       round(fps, 1),
        "ap_elephant": round(metrics.box.ap50[0], 4) if len(metrics.box.ap50) > 0 else 0,
        "ap_tiger":    round(metrics.box.ap50[1], 4) if len(metrics.box.ap50) > 1 else 0,
        "ap_leopard":  round(metrics.box.ap50[2], 4) if len(metrics.box.ap50) > 2 else 0,
        "ap_wildboar": round(metrics.box.ap50[3], 4) if len(metrics.box.ap50) > 3 else 0,
    }

    print(f"  ✅ mAP@50: {result['map50']} | Precision: {result['precision']} | Recall: {result['recall']}")
    return result


# ──────────────────────────────────────────
# PRINT COMPARISON TABLE
# ──────────────────────────────────────────

def print_table(results: list):
    header = f"\n{'='*110}\n  MODEL COMPARISON — EcoSentinel Wildlife Detection\n{'='*110}"
    row_fmt = "  {:<20} {:>8} {:>10} {:>11} {:>9} {:>8} {:>7} {:>10} {:>10}"

    lines = [
        header,
        row_fmt.format(
            "Model", "mAP@50", "mAP@50:95", "Precision", "Recall",
            "Params", "FPS", "AP Elephant", "AP WildBoar"
        ),
        "  " + "-"*106,
    ]

    for r in sorted(results, key=lambda x: -x["map50"]):
        marker = " ◄ OURS" if "Ours" in r["model"] else ""
        lines.append(row_fmt.format(
            r["model"] + marker,
            r["map50"],
            r["map5095"],
            r["precision"],
            r["recall"],
            f"{r['params_m']}M",
            r["fps"],
            r["ap_elephant"],
            r["ap_wildboar"],
        ))

    lines.append("  " + "="*106)
    table_str = "\n".join(lines)
    print(table_str)
    return table_str


# ──────────────────────────────────────────
# SAVE RESULTS
# ──────────────────────────────────────────

def save_results(results: list, table_str: str):
    # CSV
    csv_path = OUTPUT_DIR / "comparison_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  ✅ CSV saved → {csv_path}")

    # Text table
    table_path = OUTPUT_DIR / "comparison_table.txt"
    with open(table_path, "w") as f:
        f.write(table_str)
    print(f"  ✅ Table saved → {table_path}")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  🦁 EcoSentinel — Model Comparison Benchmark")
    print(f"  Device: {'GPU' if DEVICE == 0 else 'CPU'}")
    print(f"  Models: {len(MODELS)}")
    print(f"  Epochs per model: {TRAIN_EPOCHS}")
    print(f"{'='*55}")

    print("\n⚠️  NOTE: This will train each model for 50 epochs.")
    print(f"   Estimated time: ~{len(MODELS) * 35} minutes on RTX 2050")
    print("   Go get a coffee ☕\n")

    all_results = []

    for model_cfg in MODELS:
        name = model_cfg["name"]

        if model_cfg["type"] == "pretrained":
            # Already trained — just evaluate
            print(f"\n  ⏭️  {name} — using existing weights (already trained)")
            weights = model_cfg["weights"]
        else:
            # Train from scratch on your dataset
            weights = train_model(model_cfg)

        # Evaluate on test set
        result = evaluate_model(weights, name)
        all_results.append(result)

        # Print intermediate results
        print(f"  Result: mAP@50={result['map50']}, mAP@50:95={result['map5095']}")

    # Final comparison table
    table_str = print_table(all_results)
    save_results(all_results, table_str)

    print(f"\n{'='*55}")
    print(f"  🎉 Comparison Complete!")
    print(f"  Results at: {OUTPUT_DIR}")
    print(f"{'='*55}\n")
