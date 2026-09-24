
"""
Exploratory data analysis for the Dragon Fruit disease classification dataset.

Checks:
1. Class distribution per split (train/val/test) — confirms the imbalance
2. Possible data leakage from augmentation — flags likely duplicate/near-duplicate
   base images appearing across splits (common when augmentation runs BEFORE
   the split, rather than only on the train set after splitting)
3. A quick visual sample grid, one image per class

Edit DATA_ROOT to point at your dataset folder. Expected structure:

DATA_ROOT/
    train/
        class_a/*.jpg
        class_b/*.jpg
        ...
    val/
        class_a/*.jpg
        ...
    test/
        class_a/*.jpg
        ...

If your folder names differ (e.g. "training"/"validation"), adjust SPLIT_DIRS below.
"""

import os
from pathlib import Path
from collections import defaultdict
import hashlib

import matplotlib.pyplot as plt
from PIL import Image

# ---- CONFIG: edit this ----
DATA_ROOT = Path("archive/cleaned")
SPLIT_DIRS = {"train": "train", "val": "val", "test": "test"}
# ----------------------------


def list_classes(split_path: Path):
    return sorted([d.name for d in split_path.iterdir() if d.is_dir()])


def class_distribution():
    """Print image counts per class, per split."""
    counts = defaultdict(dict)
    for split_name, folder in SPLIT_DIRS.items():
        split_path = DATA_ROOT / folder
        if not split_path.exists():
            print(f"WARNING: {split_path} does not exist — skipping")
            continue
        for cls in list_classes(split_path):
            cls_path = split_path / cls
            n = len(list(cls_path.glob("*")))
            counts[cls][split_name] = n

    print(f"{'Class':<25}" + "".join(f"{s:>10}" for s in SPLIT_DIRS))
    print("-" * (25 + 10 * len(SPLIT_DIRS)))
    for cls in sorted(counts):
        row = counts[cls]
        print(f"{cls:<25}" + "".join(f"{row.get(s, 0):>10}" for s in SPLIT_DIRS))

    return counts


def file_hash(path, chunk_size=65536):
    """Full-file hash — hashing only a byte sample (e.g. first 64KB) gives
    false positives on JPEGs, since photos from the same camera/settings
    often share identical header bytes (EXIF, quantization tables) even
    when the actual image content is completely different. Hashing the
    whole file avoids that. Won't catch augmented (rotated/flipped/
    color-shifted) duplicates — those need a perceptual hash instead."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def check_exact_duplicates_across_splits():
    """Flags exact duplicate files appearing in more than one split."""
    seen = defaultdict(list)  # hash -> list of (split, path)
    for split_name, folder in SPLIT_DIRS.items():
        split_path = DATA_ROOT / folder
        if not split_path.exists():
            continue
        for cls_dir in split_path.iterdir():
            if not cls_dir.is_dir():
                continue
            for img_path in cls_dir.glob("*"):
                h = file_hash(img_path)
                seen[h].append((split_name, img_path))

    leaks = {h: v for h, v in seen.items() if len({s for s, _ in v}) > 1}
    if leaks:
        print(f"\n⚠ Found {len(leaks)} exact-duplicate files appearing across splits.")
        for h, entries in list(leaks.items())[:5]:
            print("  Example:", entries)
        print("  This IS leakage — same file in train and val/test.")
    else:
        print("\nNo exact duplicate files found across splits (good sign).")
        print("Note: this only catches identical files. If augmentation used")
        print("rotation/flip/color-jitter, near-duplicates won't be caught by")
        print("this check — inspect a few val/test images visually against")
        print("train images from the same class to sanity-check further.")

    return leaks


def sample_grid():
    """Show one sample image per class from the train split."""
    train_path = DATA_ROOT / SPLIT_DIRS["train"]
    classes = list_classes(train_path)
    n = len(classes)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i, cls in enumerate(classes):
        cls_path = train_path / cls
        img_file = next(cls_path.glob("*"), None)
        if img_file:
            img = Image.open(img_file)
            axes[i].imshow(img)
            axes[i].set_title(cls, fontsize=10)
        axes[i].axis("off")

    for j in range(len(classes), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("sample_grid.png", dpi=120)
    print("\nSaved sample_grid.png")


if __name__ == "__main__":
    print("=== Class distribution ===")
    class_distribution()

    print("\n=== Leakage check ===")
    check_exact_duplicates_across_splits()

    print("\n=== Sample grid ===")
    sample_grid()