"""
EcoSentinel — Wildlife Dataset Merger Pipeline
===============================================
Hardcoded for Ram's exact dataset structure at D:\\projects\\ecosentinel

Datasets:
  1. Leopard.v1-leopard.yolov8          → train only | class: Leopard → leopard
  2. Tegers_New.v1i.yolov8              → train/valid/test | class: Tiger → tiger
  3. wild boar test 2.v1i.yolov8        → train only | class: pig → wild_boar
  4. WildAnimalIntrusioninUrbanAreas2   → train/valid/test | Leopard + Tiger only (discard Cheetah, Jaguar, Lion)
  5. Wildlife detection.v3i.yolov8      → train/valid/test | boar + elephant + leopard + tiger (discard bear)
  6. elephants-wz5qt                    → train/valid/test | class: elephant → elephant (2030 images)

Final Unified Classes:
  0 → elephant
  1 → tiger
  2 → leopard
  3 → wild_boar
  4 → wild_dog  (no source data yet — placeholder for future datasets)

Output: D:\\projects\\ecosentinel\\unified_dataset\\
  ├── train/images + labels
  ├── val/images   + labels
  ├── test/images  + labels
  └── data.yaml    → ready for YOLOv8 training
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────

BASE_DIR    = Path(r"D:\projects\ecosentinel")
OUTPUT_DIR  = BASE_DIR / "unified_dataset"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

random.seed(42)

# Final class IDs
UNIFIED_CLASSES = {
    "elephant":  0,
    "tiger":     1,
    "leopard":   2,
    "wild_boar": 3,
    "wild_dog":  4,
}

# ──────────────────────────────────────────
# EXACT CLASS REMAPPING — Per Dataset
# Key   = original class name (lowercase)
# Value = unified class name
# Missing from this map = DISCARD
# ──────────────────────────────────────────

DATASET_CONFIGS = [

    # ── 1. Leopard dataset ──────────────────────────────────────────────────
    {
        "name":   "leopard_main",
        "root":   BASE_DIR / "Leopard.v1-leopard.yolov8",
        "splits": {
            "train": "train",   # only train exists on disk
            "val":   None,      # yaml mentions it but folder doesn't exist
            "test":  None,
        },
        "remap": {
            "leopard": "leopard",  # class 0 → leopard
        }
    },

    # ── 2. Elephant dataset (Roboflow — 2030 images) ────────────────────────
    {
        "name":   "elephant_roboflow",
        "root":   BASE_DIR / "elephants-wz5qt",
        "splits": {
            "train": "train",
            "val":   "valid",
            "test":  "test",
        },
        "remap": {
            "elephant": "elephant",  # class 0 → elephant
        }
    },

    # ── 3. Tigers dataset ───────────────────────────────────────────────────
    {
        "name":   "tiger_main",
        "root":   BASE_DIR / "Tegers_New.v1i.yolov8",
        "splits": {
            "train": "train",
            "val":   "valid",
            "test":  "test",
        },
        "remap": {
            "tiger": "tiger",  # class 0 → tiger
        }
    },

    # ── 4. Wild Boar dataset ────────────────────────────────────────────────
    {
        "name":   "boar_main",
        "root":   BASE_DIR / "wild boar test 2.v1i.yolov8",
        "splits": {
            "train": "train",   # only train exists on disk
            "val":   None,
            "test":  None,
        },
        "remap": {
            "pig": "wild_boar",  # class 0 ("pig") → wild_boar
        }
    },

    # ── 5. Wild Animal Intrusion dataset ────────────────────────────────────
    # Original classes: ['Cheetah'=0, 'Jaguar'=1, 'Leopard'=2, 'Lion'=3, 'Tiger'=4]
    # Keep ONLY Leopard and Tiger — Cheetah, Jaguar, Lion are discarded
    {
        "name":   "intrusion_multi",
        "root":   BASE_DIR / "WildAnimalIntrusioninUrbanAreas2.v3i.yolov8",
        "splits": {
            "train": "train",
            "val":   "valid",
            "test":  "test",
        },
        "remap": {
            # "cheetah" → NOT in remap = DISCARDED
            # "jaguar"  → NOT in remap = DISCARDED
            "leopard": "leopard",  # class 2 → leopard ✅
            # "lion"    → NOT in remap = DISCARDED
            "tiger":   "tiger",    # class 4 → tiger ✅
        }
    },

    # ── 6. Wildlife Detection dataset ───────────────────────────────────────
    # Original classes: ['bear'=0, 'boar'=1, 'elephant'=2, 'leopard'=3, 'tiger'=4]
    # Discard bear only
    {
        "name":   "wildlife_5in1",
        "root":   BASE_DIR / "Wildlife detection.v3i.yolov8",
        "splits": {
            "train": "train",
            "val":   "valid",
            "test":  "test",
        },
        "remap": {
            # "bear" → NOT in remap = DISCARDED ❌
            "boar":     "wild_boar",  # class 1 → wild_boar ✅
            "elephant": "elephant",   # class 2 → elephant  ✅
            "leopard":  "leopard",    # class 3 → leopard   ✅
            "tiger":    "tiger",      # class 4 → tiger     ✅
        }
    },
]


# ──────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────

def read_yaml_classes(dataset_root: Path) -> list:
    """Read class names list from data.yaml inside a dataset folder."""
    for name in ["data.yaml", "dataset.yaml", "classes.yaml"]:
        p = dataset_root / name
        if p.exists():
            with open(p) as f:
                cfg = yaml.safe_load(f)
            names = cfg.get("names", [])
            if isinstance(names, dict):
                names = [names[i] for i in sorted(names.keys())]
            return [n.lower().strip() for n in names]
    return []


def get_pairs(split_dir: Path):
    """Get (image_path, label_path) pairs from a split directory."""
    if split_dir is None or not split_dir.exists():
        return []

    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        return []

    pairs = []
    for img in img_dir.iterdir():
        if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def remap_label(label_path: Path, original_classes: list, remap: dict):
    """
    Filter and remap annotations in a YOLO .txt label file.
    Returns list of new annotation lines, or None if nothing to keep.
    """
    try:
        lines = label_path.read_text().strip().splitlines()
    except Exception:
        return None

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        old_id = int(parts[0])
        if old_id >= len(original_classes):
            continue

        original_name = original_classes[old_id]  # already lowercased
        unified_name  = remap.get(original_name)

        if unified_name is None:
            continue  # ❌ DISCARD this annotation

        new_id = UNIFIED_CLASSES[unified_name]
        new_lines.append(f"{new_id} {' '.join(parts[1:])}")

    return new_lines if new_lines else None


# ──────────────────────────────────────────
# MERGER CLASS
# ──────────────────────────────────────────

class EcoSentinelMerger:

    def __init__(self, output_dir: Path):
        self.out = Path(output_dir)
        self.counters = {"train": 0, "val": 0, "test": 0}
        self.stats    = defaultdict(lambda: defaultdict(int))  # stats[split][class]
        self.pending  = []  # train-only pairs to re-split later

        for split in ["train", "val", "test"]:
            (self.out / split / "images").mkdir(parents=True, exist_ok=True)
            (self.out / split / "labels").mkdir(parents=True, exist_ok=True)

    def _save(self, img: Path, lines: list, split: str, prefix: str):
        n   = self.counters[split]
        ext = img.suffix.lower()
        dst_img = self.out / split / "images" / f"{prefix}_{n:06d}{ext}"
        dst_lbl = self.out / split / "labels" / f"{prefix}_{n:06d}.txt"

        shutil.copy2(img, dst_img)
        dst_lbl.write_text("\n".join(lines))

        for line in lines:
            cid  = int(line.split()[0])
            cname = [k for k, v in UNIFIED_CLASSES.items() if v == cid][0]
            self.stats[split][cname] += 1

        self.counters[split] += 1

    def process_dataset(self, cfg: dict):
        root    = cfg["root"]
        name    = cfg["name"]
        remap   = {k.lower(): v for k, v in cfg["remap"].items()}
        splits  = cfg["splits"]

        print(f"\n{'─'*55}")
        print(f"  📂 {name}")
        print(f"  📍 {root}")

        if not root.exists():
            print(f"  ❌ Folder not found — skipping")
            return

        original_classes = read_yaml_classes(root)
        if not original_classes:
            print(f"  ❌ Could not read class names from data.yaml — skipping")
            return

        print(f"  📋 Original classes: {original_classes}")
        print(f"  🔀 Keeping: {list(remap.keys())}")

        has_val  = splits.get("val")  is not None
        has_test = splits.get("test") is not None

        if not has_val and not has_test:
            # Only train — pool for re-split
            split_dir = root / splits["train"]
            pairs     = get_pairs(split_dir)
            print(f"  ⚠️  Train-only ({len(pairs)} pairs) → will re-split 80/10/10")
            self.pending.extend(
                (img, lbl, original_classes, remap, name) for img, lbl in pairs
            )
        else:
            # Has existing splits — use them
            for split_key, folder_name in splits.items():
                if folder_name is None:
                    continue
                split_dir = root / folder_name
                pairs     = get_pairs(split_dir)
                if not pairs:
                    continue

                kept = dropped = 0
                for img, lbl in pairs:
                    lines = remap_label(lbl, original_classes, remap)
                    if lines:
                        self._save(img, lines, split_key, name)
                        kept += 1
                    else:
                        dropped += 1

                print(f"  ✅ {split_key:5s}: {kept:4d} kept  |  {dropped:4d} dropped")

    def finalize_pending(self):
        if not self.pending:
            return

        print(f"\n{'─'*55}")
        print(f"  🔀 Re-splitting {len(self.pending)} train-only samples → 80/10/10")

        random.shuffle(self.pending)
        n       = len(self.pending)
        n_val   = int(n * VAL_RATIO)
        n_test  = int(n * TEST_RATIO)

        buckets = (
            [("val",   item) for item in self.pending[:n_val]] +
            [("test",  item) for item in self.pending[n_val:n_val + n_test]] +
            [("train", item) for item in self.pending[n_val + n_test:]]
        )

        kept = dropped = 0
        for split_key, (img, lbl, orig_cls, remap, name) in buckets:
            lines = remap_label(lbl, orig_cls, remap)
            if lines:
                self._save(img, lines, split_key, name)
                kept += 1
            else:
                dropped += 1

        print(f"  ✅ Kept: {kept}  |  Dropped: {dropped}")

    def write_yaml(self):
        cfg = {
            "path":  str(self.out.resolve()),
            "train": "train/images",
            "val":   "val/images",
            "test":  "test/images",
            "nc":    len(UNIFIED_CLASSES),
            "names": {v: k for k, v in UNIFIED_CLASSES.items()},
        }
        out_path = self.out / "data.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"\n  ✅ data.yaml → {out_path}")

    def print_report(self):
        print(f"\n{'═'*55}")
        print(f"  📊 FINAL DATASET REPORT")
        print(f"{'═'*55}")

        total = 0
        for split in ["train", "val", "test"]:
            n = self.counters[split]
            total += n
            print(f"\n  [{split.upper()}] — {n} images")
            for cls, count in sorted(self.stats[split].items(), key=lambda x: x[0]):
                bar = "█" * min(count // 30, 30)
                print(f"    {cls:<12} {count:>5}  {bar}")

        print(f"\n  TOTAL IMAGES : {total}")

        print(f"\n{'─'*55}")
        print(f"  🏥 CLASS HEALTH CHECK (train set)")
        print(f"{'─'*55}")
        for cls in sorted(UNIFIED_CLASSES.keys()):
            count = self.stats["train"].get(cls, 0)
            if count == 0:
                status = "🔴 NO DATA — find more datasets!"
            elif count < 300:
                status = f"🟡 SPARSE ({count}) — run augmentation"
            elif count < 1000:
                status = f"🟠 MODERATE ({count}) — acceptable"
            else:
                status = f"🟢 GOOD ({count})"
            print(f"    {cls:<12}  {status}")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'═'*55}")
    print(f"  🦁 EcoSentinel Dataset Pipeline")
    print(f"  Output → {OUTPUT_DIR}")
    print(f"{'═'*55}")

    merger = EcoSentinelMerger(OUTPUT_DIR)

    for cfg in DATASET_CONFIGS:
        merger.process_dataset(cfg)

    merger.finalize_pending()
    merger.write_yaml()
    merger.print_report()

    print(f"\n{'═'*55}")
    print(f"  🎉 Done! Dataset ready at:")
    print(f"  {OUTPUT_DIR}")
    print(f"\n  Next step — start training:")
    print(f"  yolo train model=yolov8m.pt data={OUTPUT_DIR}\\data.yaml epochs=100 imgsz=640")
    print(f"{'═'*55}\n")
