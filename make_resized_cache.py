"""
make_resized_cache.py -- build the resized JPEG copies of the dataset that the reported runs trained on.

Reading full-size photos for every epoch is slow, so the experiments used two caches, each a
copy of the dataset with the same train/val/test/<class> layout:

  small   longest side <= 160 px, JPEG quality 92   -> used by the 128 px runs (arms A, C, D, E, baseline)
  med     longest side <= 256 px, JPEG quality 95   -> used by the 224 px run (arm B)

The training script then resizes each cached image to exactly img_size x img_size.
Aspect ratio is kept here (thumbnail) and squashed there, as in the original runs.

Usage
  python make_resized_cache.py --src Cleaned --dst small --size 160 --quality 92
  python make_resized_cache.py --src Cleaned --dst med   --size 256 --quality 95
"""
import argparse
import os

from PIL import Image

EXTENSIONS = (".jpg", ".jpeg", ".png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="dataset root (train/ val/ test/ with one folder per class)")
    p.add_argument("--dst", required=True, help="output folder (created; must not already contain images)")
    p.add_argument("--size", type=int, required=True, help="longest side after resizing, in pixels")
    p.add_argument("--quality", type=int, default=92, help="JPEG quality")
    a = p.parse_args()

    if os.path.abspath(a.src) == os.path.abspath(a.dst):
        raise SystemExit("--dst must differ from --src")
    n = 0
    for root, _, files in os.walk(a.src):
        for f in sorted(files):
            if not f.lower().endswith(EXTENSIONS):
                continue
            out_dir = root.replace(a.src, a.dst, 1)
            os.makedirs(out_dir, exist_ok=True)
            im = Image.open(os.path.join(root, f)).convert("RGB")
            im.thumbnail((a.size, a.size))
            im.save(os.path.join(out_dir, os.path.splitext(f)[0] + ".jpg"), quality=a.quality)
            n += 1
    print(f"wrote {n} images to {a.dst}")


if __name__ == "__main__":
    main()
