"""
EcoSentinel — Real-Time Wildlife Detection
==========================================
Runs inference on:
  - Live webcam / IP camera
  - Video files (mp4, avi, etc.)
  - Single images
  - Folder of images

Features:
  - Bounding boxes with class name + confidence
  - Per-class color coding
  - FPS counter
  - Alert system — logs detections with timestamp
  - Auto-save snapshots when dangerous animal detected
  - Confidence threshold tunable per class

Classes:
  0 → elephant
  1 → tiger
  2 → leopard
  3 → wild_boar
  4 → wild_dog

Requirements:
  pip install ultralytics opencv-python cvzone

Run:
  # Webcam
  python inference.py --source 0

  # Video file
  python inference.py --source path/to/video.mp4

  # Image
  python inference.py --source path/to/image.jpg

  # IP Camera (RTSP)
  python inference.py --source rtsp://username:password@ip:port/stream

  # Folder of images
  python inference.py --source path/to/folder/
"""

import cv2
import argparse
import time
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO

# ──────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────

MODEL_PATH = r"D:\projects\ecosentinel\runs\ecosentinel_v1\weights\best.pt"

# Output directory for saved snapshots and logs
OUTPUT_DIR = Path(r"D:\projects\ecosentinel\detections")

# Confidence thresholds per class
# Lower = more sensitive (more detections, more false alarms)
# Higher = more strict (fewer detections, fewer false alarms)
# Recommendation for intrusion alert system: 0.35-0.45
CONF_THRESHOLD = {
    "elephant": 0.35,   # lower — sparse training data, be more sensitive
    "tiger":    0.40,
    "leopard":  0.40,
    "wild_boar":0.45,   # higher — well trained, be strict
    "wild_dog": 0.35,
}
DEFAULT_CONF = 0.40     # fallback if class not in above dict

# Classes that trigger an ALERT snapshot save
# Set to all 5 if you want every detection saved
ALERT_CLASSES = {"elephant", "tiger", "leopard", "wild_dog"}  # boar excluded as less dangerous

# Per-class colors (BGR format for OpenCV)
CLASS_COLORS = {
    "elephant": (255, 140, 0),    # orange
    "tiger":    (0, 69, 255),     # red-orange
    "leopard":  (0, 215, 255),    # gold
    "wild_boar":(180, 105, 255),  # pink
    "wild_dog": (0, 255, 127),    # spring green
    "default":  (200, 200, 200),  # grey for unknown
}

# Alert cooldown — don't save snapshot more than once every N seconds per class
# Prevents flooding your disk with images of the same animal
ALERT_COOLDOWN_SEC = 10

# Display settings
SHOW_FPS       = True
SHOW_TIMESTAMP = True
SHOW_CONF      = True
FONT           = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────────────
# DETECTION LOGGER
# ──────────────────────────────────────────────────────

class DetectionLogger:
    def __init__(self, output_dir: Path):
        self.output_dir   = output_dir
        self.snapshot_dir = output_dir / "snapshots"
        self.log_path     = output_dir / "detections_log.csv"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # CSV log
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "class", "confidence", "bbox_xyxy", "snapshot"])

        # Cooldown tracker — last alert time per class
        self.last_alert = defaultdict(lambda: 0)

        # Session stats
        self.session_counts = defaultdict(int)

    def log(self, class_name: str, confidence: float, bbox, frame):
        now       = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshot_name = ""

        # Check cooldown
        if class_name in ALERT_CLASSES:
            if now - self.last_alert[class_name] >= ALERT_COOLDOWN_SEC:
                # Save snapshot
                snapshot_name = f"{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                snapshot_path = self.snapshot_dir / snapshot_name
                cv2.imwrite(str(snapshot_path), frame)
                self.last_alert[class_name] = now
                print(f"  🚨 ALERT: {class_name} detected! Snapshot → {snapshot_path}")

        # Write to CSV
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, class_name, f"{confidence:.3f}", str(bbox), snapshot_name])

        self.session_counts[class_name] += 1

    def print_session_summary(self):
        print(f"\n{'═'*50}")
        print(f"  📊 SESSION SUMMARY")
        print(f"{'═'*50}")
        total = sum(self.session_counts.values())
        if total == 0:
            print("  No detections this session.")
        else:
            for cls, count in sorted(self.session_counts.items(), key=lambda x: -x[1]):
                print(f"  {cls:<15} {count:>5} detections")
            print(f"  {'TOTAL':<15} {total:>5}")
        print(f"\n  Log saved to : {self.log_path}")
        print(f"  Snapshots at : {self.snapshot_dir}")
        print(f"{'═'*50}\n")


# ──────────────────────────────────────────────────────
# DRAWING HELPERS
# ──────────────────────────────────────────────────────

def draw_detection(frame, box, class_name, confidence):
    """Draw bounding box + label on frame."""
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS.get(class_name, CLASS_COLORS["default"])

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    label = f"{class_name}"
    if SHOW_CONF:
        label += f" {confidence:.0%}"

    (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)

    # Label text
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return frame


def draw_overlay(frame, fps, detection_counts):
    """Draw FPS + timestamp + detection summary overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # FPS
    if SHOW_FPS:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24),
                    FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Timestamp
    if SHOW_TIMESTAMP:
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w - 90, 24),
                    FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Detection counts (bottom left)
    if detection_counts:
        y_offset = h - 10
        for cls, count in sorted(detection_counts.items()):
            color = CLASS_COLORS.get(cls, CLASS_COLORS["default"])
            cv2.putText(frame, f"{cls}: {count}", (10, y_offset),
                        FONT, 0.55, color, 2, cv2.LINE_AA)
            y_offset -= 22

    return frame


# ──────────────────────────────────────────────────────
# MAIN INFERENCE LOOP
# ──────────────────────────────────────────────────────

def run_inference(source, save_video=False):
    """
    Main inference loop.
    source: 0 (webcam) | "path/to/video.mp4" | "path/to/image.jpg" | "rtsp://..."
    """
    print(f"\n{'═'*50}")
    print(f"  🦁 EcoSentinel — Wildlife Detection")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Source : {source}")
    print(f"{'═'*50}")
    print(f"  Press Q to quit | Press S to save snapshot manually")
    print(f"{'═'*50}\n")

    # Load model
    model  = YOLO(MODEL_PATH)
    logger = DetectionLogger(OUTPUT_DIR)

    # Open source
    is_image = isinstance(source, str) and Path(source).suffix.lower() in \
               [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    if is_image:
        run_image_inference(model, logger, source)
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Could not open source: {source}")
        return

    # Get video properties
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  📹 Source: {w}x{h} @ {fps_src:.1f}fps")

    # Video writer (optional)
    writer = None
    if save_video:
        out_path = OUTPUT_DIR / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps_src, (w, h))
        print(f"  💾 Saving output video → {out_path}")

    # FPS tracking
    fps_buffer = deque(maxlen=30)
    prev_time  = time.time()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  ✅ End of stream.")
            break

        frame_count += 1

        # Run detection
        results = model(frame, verbose=False)[0]

        # Process detections
        detection_counts = defaultdict(int)

        for box in results.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            bbox       = box.xyxy[0].tolist()

            # Apply per-class threshold
            threshold = CONF_THRESHOLD.get(class_name, DEFAULT_CONF)
            if confidence < threshold:
                continue

            # Draw detection
            draw_detection(frame, bbox, class_name, confidence)
            detection_counts[class_name] += 1

            # Log detection
            logger.log(class_name, confidence, bbox, frame)

        # FPS calculation
        curr_time = time.time()
        fps_buffer.append(1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time
        fps = sum(fps_buffer) / len(fps_buffer)

        # Draw overlay
        draw_overlay(frame, fps, detection_counts)

        # Show frame
        cv2.imshow("EcoSentinel — Wildlife Detection (Q to quit)", frame)

        # Save frame to video
        if writer:
            writer.write(frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n  👋 Quit by user.")
            break
        elif key == ord("s"):
            snap = OUTPUT_DIR / "snapshots" / f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(snap), frame)
            print(f"  📸 Manual snapshot saved → {snap}")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    logger.print_session_summary()


def run_image_inference(model, logger, image_path: str):
    """Run inference on a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not read image: {image_path}")
        return

    results = model(frame, verbose=False)[0]

    detection_counts = defaultdict(int)
    for box in results.boxes:
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        bbox       = box.xyxy[0].tolist()

        threshold = CONF_THRESHOLD.get(class_name, DEFAULT_CONF)
        if confidence < threshold:
            continue

        draw_detection(frame, bbox, class_name, confidence)
        detection_counts[class_name] += 1
        logger.log(class_name, confidence, bbox, frame)

    # Print results
    print(f"\n  Detections in {Path(image_path).name}:")
    if detection_counts:
        for cls, count in detection_counts.items():
            print(f"    🔍 {cls}: {count} instance(s)")
    else:
        print(f"    ✅ No target animals detected")

    # Save result image
    out_path = OUTPUT_DIR / f"result_{Path(image_path).stem}.jpg"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"\n  💾 Result saved → {out_path}")

    # Show image
    cv2.imshow(f"EcoSentinel — {Path(image_path).name} (any key to close)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    logger.print_session_summary()


# ──────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="EcoSentinel Wildlife Detection")
    parser.add_argument(
        "--source", default="0",
        help="Source: 0=webcam | path/to/video.mp4 | path/to/image.jpg | rtsp://..."
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Save output video with detections drawn"
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Override confidence threshold for all classes (e.g. 0.4)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override conf if provided
    if args.conf is not None:
        for k in CONF_THRESHOLD:
            CONF_THRESHOLD[k] = args.conf
        DEFAULT_CONF = args.conf
        print(f"  ⚙️  Confidence override: {args.conf}")

    # Convert source to int if webcam
    source = args.source
    if source.isdigit():
        source = int(source)

    run_inference(source, save_video=args.save_video)
