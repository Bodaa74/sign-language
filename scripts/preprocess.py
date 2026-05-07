"""
ASL Sign Language Translator - Data Preprocessing Pipeline
===========================================================
Handles dataset download hints, image preprocessing, augmentation,
train/val/test splitting, and class imbalance correction.

Supported Datasets:
  - Kaggle ASL Alphabet (87,000 images, 29 classes A–Z + space/del/nothing)
  - ASL Fingerspelling Dataset (MNIST-style, grayscale)

Usage:
  python scripts/preprocess.py --data_dir data/raw --output_dir data/processed
"""

import os
import cv2
import numpy as np
import argparse
import shutil
import random
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
IMG_SIZE = 224          # MobileNetV2 / ResNet50 input size
CHANNELS = 3
SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
MIN_SAMPLES_PER_CLASS = 1000   # Upsample classes below this threshold

AUGMENT_CONFIG = dict(
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.10,
    zoom_range=0.15,
    horizontal_flip=False,     # ASL is hand-specific, no flipping
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_image(path: str) -> np.ndarray | None:
    """Load, resize and normalize a single image. Returns None on failure."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0   # Normalize [0, 1]
    return img


def save_image(img: np.ndarray, path: str):
    """Save normalized float image back to disk."""
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


# ─────────────────────────────────────────────
# Dataset Scanner
# ─────────────────────────────────────────────

def scan_dataset(data_dir: str) -> dict[str, list[str]]:
    """
    Expects folder structure:
        data/raw/
            A/  img1.jpg  img2.jpg ...
            B/  ...
            ...
            Z/
    Returns a dict: {class_name: [file_paths]}
    """
    data_dir = Path(data_dir)
    class_map: dict[str, list[str]] = {}

    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name.upper()
        if label not in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            continue   # Skip non-letter folders (space, del, nothing)
        files = [
            str(f) for f in cls_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if files:
            class_map[label] = files

    print(f"\n📂 Found {len(class_map)} classes:")
    for k, v in sorted(class_map.items()):
        print(f"   {k}: {len(v):,} images")
    return class_map


# ─────────────────────────────────────────────
# Class Imbalance Correction
# ─────────────────────────────────────────────

def balance_classes(
    class_map: dict[str, list[str]],
    output_dir: str,
    min_samples: int = MIN_SAMPLES_PER_CLASS,
) -> dict[str, list[str]]:
    """
    For classes with fewer than min_samples images, generate augmented
    copies until the threshold is reached.
    """
    aug_dir = Path(output_dir) / "augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)

    datagen = ImageDataGenerator(**AUGMENT_CONFIG)
    balanced: dict[str, list[str]] = {}

    for label, paths in class_map.items():
        label_dir = aug_dir / label
        label_dir.mkdir(exist_ok=True)

        current_paths = list(paths)
        deficit = min_samples - len(current_paths)

        if deficit > 0:
            print(f"   ↑ {label}: augmenting {deficit} extra images …")
            aug_count = 0
            while aug_count < deficit:
                src_path = random.choice(paths)
                img = load_image(src_path)
                if img is None:
                    continue
                img_4d = img[np.newaxis, ...]   # (1, H, W, C)
                for batch in datagen.flow(img_4d, batch_size=1):
                    aug_img = batch[0]
                    out_path = str(label_dir / f"aug_{aug_count:05d}.jpg")
                    save_image(aug_img, out_path)
                    current_paths.append(out_path)
                    aug_count += 1
                    if aug_count >= deficit:
                        break

        balanced[label] = current_paths

    return balanced


# ─────────────────────────────────────────────
# Split & Save
# ─────────────────────────────────────────────

def split_and_save(
    class_map: dict[str, list[str]],
    output_dir: str,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
):
    """
    Splits each class into train / val / test and copies files
    into the output directory with the expected structure:
        processed/
            train/A/ train/B/ ...
            val/A/   val/B/ ...
            test/A/  test/B/ ...
    """
    output_dir = Path(output_dir)
    splits = {"train": [], "val": [], "test": []}
    all_labels = []

    for label, paths in sorted(class_map.items()):
        random.shuffle(paths)
        n = len(paths)
        n_test = max(1, int(n * test_split))
        n_val  = max(1, int(n * val_split))

        test_set  = paths[:n_test]
        val_set   = paths[n_test:n_test + n_val]
        train_set = paths[n_test + n_val:]

        for split_name, subset in [("train", train_set), ("val", val_set), ("test", test_set)]:
            dest_dir = output_dir / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in subset:
                dst = dest_dir / Path(src).name
                shutil.copy2(src, dst)
                splits[split_name].append((str(dst), label))

        all_labels.extend([label] * n)

    # Report
    for split_name, items in splits.items():
        cnt = Counter(lbl for _, lbl in items)
        print(f"\n📊 {split_name.upper()} split: {len(items):,} total")
        for lbl in sorted(cnt):
            print(f"   {lbl}: {cnt[lbl]:,}")

    return splits


# ─────────────────────────────────────────────
# TFRecord Writer (optional, for large datasets)
# ─────────────────────────────────────────────

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecord(split_items: list[tuple[str, str]], label_map: dict[str, int], out_path: str):
    """Write a split to a TFRecord file for efficient training."""
    with tf.io.TFRecordWriter(out_path) as writer:
        for img_path, label in tqdm(split_items, desc=f"Writing {out_path}"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            _, buf = cv2.imencode(".jpg", img)
            feature = {
                "image": _bytes_feature(buf.tobytes()),
                "label": _int64_feature(label_map[label]),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    set_seed()

    print("=" * 60)
    print("  ASL Translator — Preprocessing Pipeline")
    print("=" * 60)

    # 1. Scan
    class_map = scan_dataset(args.data_dir)
    assert class_map, "No classes found. Check --data_dir path."

    # 2. Balance
    print("\n🔧 Balancing classes …")
    class_map = balance_classes(class_map, args.output_dir)

    # 3. Split & save
    print("\n✂️  Splitting dataset …")
    splits = split_and_save(class_map, args.output_dir)

    # 4. (Optional) Write TFRecords
    if args.tfrecord:
        label_map = {lbl: i for i, lbl in enumerate(sorted(class_map))}
        for split_name, items in splits.items():
            tfr_path = os.path.join(args.output_dir, f"{split_name}.tfrecord")
            write_tfrecord(items, label_map, tfr_path)

    print("\n✅ Preprocessing complete!")
    print(f"   Output: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Dataset Preprocessor")
    parser.add_argument("--data_dir",    default="data/raw",       help="Raw dataset root")
    parser.add_argument("--output_dir",  default="data/processed",  help="Output directory")
    parser.add_argument("--tfrecord",    action="store_true",        help="Also write TFRecords")
    args = parser.parse_args()
    main(args)
