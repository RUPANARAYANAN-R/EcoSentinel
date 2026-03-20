"""
EcoSentinel — Hard Negative Extractor
=======================================
Takes images that were DROPPED by the pipeline (contain cows, dogs, deer,
cars, people etc.) and adds them as background/negative samples.

This teaches the model what NOT to detect — critical for preventing
false positives in real deployment.

Sources of negatives used here:
  1. elephants-wz5qt dataset → dropped images (cow, dog, deer, buffalo etc.)
  2. Wildlife detection dataset → dropped images (bear only)
  3. WildAnimalIntrusion dataset → dropped images (cheetah, jaguar, lion)

Empty .txt label file = "no target animals in this image" = negative sample.

Run AFTER ecosentinel_pipeline.py has already built the unified_dataset.
"""

import shutil
import random
from pathlib import Path

random.seed(42)

BASE_DIR    = Path(r"D:\projects\ecosentinel")
OUTPUT_DIR  = BASE_DIR / "unified_dataset"

# How many negatives to add per split
# Too many negatives = model becomes overly conservative (misses real animals)
# Too few           = model hallucinates on non-target animals
# Rule of thumb: ~10-15% of your total training images
MAX_NEGATIVES = {
    "train": 400,
    "val":   50,
    "test":  50,
}

# ─────────────────────────────────────────────────────
# NEGATIVE SOURCES
# Each entry: (dataset_folder, split_folder_name)
# We extract images from these that have NO target animal annotations
# ─────────────────────────────────────────────────────

NEGATIVE_SOURCES = [

    # Best source — 1459 dropped images (cow, dog, deer, buffalo, car, person etc.)
    # These are the most valuable confusers for your use case
    {
        "name":  "elephant_dataset_negatives",
        "root":  BASE_DIR / "elephants-wz5qt",
        "splits": {
            "train": "train",
            "val":   "valid",
            "test":  "test",
        },
        # Classes we want to KEEP as negatives (confusers similar to target animals)
        # Anything in this list = image is a useful negative
        # We skip classes too different from wildlife (car, clock, lights etc.)
        "useful_confusers": [
            "bear", "buffalo", "cow", "deer", "doe", "dog",
            "elk", "fox", "roe", "wild-boar",  # already covered but as negatives too
            "person",   # important negative — detect animals but NOT people
            "southern-boobook",
        ],
        # If None → take ALL dropped images regardless of what they contain
        # Set to None if you want maximum negatives
        "filter_to_confusers": False,  # take all dropped images
    },

    # Bear images from Wildlife Detection — bear looks similar to boar/leopard
    {
        "name":  "wildlife_bear_negatives",
        "root":  BASE_DIR / "Wildlife detection.v3i.yolov8",
        "splits": {
            "train": "train",
            "val":   "valid",
            "test":  "test",
        },
        "filter_to_confusers": False,
    },

    # Cheetah, Jaguar, Lion from WildAnimalIntrusion
    # These are the hardest confusers — big cats that look like leopard/tiger
    # Model MUST learn these are NOT leopard/tiger
    {
        "name":  "intrusion_negatives",
        "root":  BASE_DIR / "WildAnimalIntrusioninUrbanAreas2.v3i.yolov8",
        "splits": {
            "train": "train",
            "val":   "valid",
            "test":  "test",
        },
        "filter_to_confusers": False,
    },
]

# Target class names — images containing ONLY these are negatives
TARGET_CLASSES = {"elephant", "tiger", "leopard", "wild_boar", "wild_dog"}


def read_yaml_classes(dataset_root: Path) -> list:
    import yaml
    for name in ["data.yaml", "dataset.yaml"]:
        p = dataset_root / name
        if p.exists():
            with open(p) as f:
                cfg = yaml.safe_load(f)
            names = cfg.get("names", [])
            if isinstance(names, dict):
                names = [names[i] for i in sorted(names.keys())]
            return [n.lower().strip() for n in names]
    return []


def image_has_only_non_targets(label_path: Path, all_classes: list) -> bool:
    """
    Returns True if the label file contains NO target animal annotations.
    i.e. this image is a valid negative sample.
    """
    if not label_path.exists():
        return True  # no label = background image = perfect negative

    try:
        lines = label_path.read_text().strip().splitlines()
    except Exception:
        return False

    if not lines:
        return True  # empty label = background

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 1:
            continue
        class_id = int(parts[0])
        if class_id < len(all_classes):
            cls_name = all_classes[class_id].lower()
            if cls_name in TARGET_CLASSES:
                return False  # contains a target animal → NOT a negative

    return True  # all annotations are non-target → valid negative


def collect_negatives(source: dict) -> dict:
    """Collect negative image paths from a dataset source."""
    root       = source["root"]
    splits_cfg = source["splits"]

    if not root.exists():
        print(f"  ⚠️  {source['name']}: folder not found, skipping")
        return {"train": [], "val": [], "test": []}

    all_classes = read_yaml_classes(root)
    if not all_classes:
        print(f"  ⚠️  {source['name']}: no data.yaml found, skipping")
        return {"train": [], "val": [], "test": []}

    result = {"train": [], "val": [], "test": []}

    for split_key, folder_name in splits_cfg.items():
        if folder_name is None:
            continue

        split_dir = root / folder_name
        img_dir   = split_dir / "images"
        lbl_dir   = split_dir / "labels"

        if not img_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue

            lbl_file = lbl_dir / (img_file.stem + ".txt") if lbl_dir.exists() else None

            if image_has_only_non_targets(lbl_file, all_classes):
                result[split_key].append(img_file)

    return result


def add_negatives_to_dataset():
    print("\n" + "═"*55)
    print("  🎯 Hard Negative Extractor")
    print("  Teaching the model what NOT to detect")
    print("═"*55)

    # Collect all negatives from all sources
    all_negatives = {"train": [], "val": [], "test": []}

    for source in NEGATIVE_SOURCES:
        print(f"\n  📂 {source['name']}")
        negatives = collect_negatives(source)
        for split in ["train", "val", "test"]:
            count = len(negatives[split])
            all_negatives[split].extend(negatives[split])
            print(f"    {split}: {count} negatives found")

    print(f"\n  📊 Total negatives collected:")
    for split in ["train", "val", "test"]:
        print(f"    {split}: {len(all_negatives[split])}")

    # Shuffle and cap at MAX_NEGATIVES
    added = {"train": 0, "val": 0, "test": 0}
    counter = {}

    # Get current image count in output for naming
    for split in ["train", "val", "test"]:
        img_dir = OUTPUT_DIR / split / "images"
        counter[split] = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0

    for split in ["train", "val", "test"]:
        negatives = all_negatives[split]
        random.shuffle(negatives)
        negatives = negatives[:MAX_NEGATIVES[split]]  # cap

        out_img_dir = OUTPUT_DIR / split / "images"
        out_lbl_dir = OUTPUT_DIR / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in negatives:
            n   = counter[split]
            ext = img_path.suffix.lower()
            dst_img = out_img_dir / f"neg_{split}_{n:06d}{ext}"
            dst_lbl = out_lbl_dir / f"neg_{split}_{n:06d}.txt"

            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("")  # empty label = no targets = negative sample

            counter[split] += 1
            added[split]   += 1

    print(f"\n  ✅ Negatives added to unified_dataset:")
    print(f"    train: +{added['train']} images (empty labels)")
    print(f"    val  : +{added['val']}   images (empty labels)")
    print(f"    test : +{added['test']}  images (empty labels)")

    print(f"\n  🏁 Updated dataset totals:")
    for split in ["train", "val", "test"]:
        img_dir = OUTPUT_DIR / split / "images"
        total   = len(list(img_dir.glob("*.*")))
        print(f"    {split}: {total} images total")

    print("\n" + "═"*55)
    print("  ✅ Done! Re-run train_ecosentinel.py to train with negatives.")
    print("═"*55 + "\n")


if __name__ == "__main__":
    add_negatives_to_dataset()
