"""
arm_figures.py -- one summary figure per experiment arm, built from the finished runs.

For each arm it plots (3 panels):
  1. validation macro-F1 per epoch, one faint line per seed plus the mean
  2. train and validation loss, mean over seeds
  3. test confusion matrix summed over seeds (cell = count, % = share of the true class)

It reads outputs/<run>/{results.json, history.json, test_report.txt} written by
train_transfer.py and saves outputs/_summary/arm_<X>.png.

Arms (run name = <prefix>_s<seed>, seeds 0-4):
  A  pretrained ResNet18, 128 px, lr 1e-4   tl_pre_128
  B  pretrained ResNet18, 224 px, lr 1e-4   tl_pre_224
  C  scratch   ResNet18, 128 px, lr 1e-3    tl_scratch_128
  D  pretrained ResNet18, 128 px, lr 1e-3   tl_pre_128_lr1e3
  E  scratch   ResNet18, 128 px, lr 1e-4    tl_scratch_128_lr1e4

Usage
  python arm_figures.py --out_root outputs --arms B E
"""
import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = {
    "A": "tl_pre_128",
    "B": "tl_pre_224",
    "C": "tl_scratch_128",
    "D": "tl_pre_128_lr1e3",
    "E": "tl_scratch_128_lr1e4",
}
CLASSES = ["Bad fruit", "Bad leaf", "Good fruit", "Good leaf"]
MARKER = "confusion matrix (rows true, cols predicted):"


def load_cm(path):
    """The matrix is printed at the end of test_report.txt: 4 classes -> 16 integers."""
    txt = open(path).read()
    nums = [int(x) for x in re.findall(r"\d+", txt.split(MARKER)[1])]
    assert len(nums) == 16, f"expected 16 integers in the confusion matrix of {path}, found {len(nums)}"
    return np.array(nums).reshape(4, 4)


def arm_figure(arm, out_root, n_seeds):
    prefix = ARMS[arm]
    runs = [f"{prefix}_s{s}" for s in range(n_seeds)
            if os.path.exists(os.path.join(out_root, f"{prefix}_s{s}", "results.json"))]
    if not runs:
        print(f"arm {arm}: no finished runs under {out_root} (looked for {prefix}_s0..s{n_seeds - 1})")
        return None
    hist = [json.load(open(os.path.join(out_root, r, "history.json"))) for r in runs]
    cm = sum(load_cm(os.path.join(out_root, r, "test_report.txt")) for r in runs)
    rows = cm / cm.sum(1, keepdims=True)
    ep = np.arange(1, len(hist[0]["val_f1"]) + 1)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for h in hist:
        ax[0].plot(ep, h["val_f1"], color="tab:blue", alpha=0.25)
    ax[0].plot(ep, np.mean([h["val_f1"] for h in hist], 0), color="tab:blue", lw=2.5, label="mean")
    ax[0].set(title=f"Arm {arm}: val macro-F1, one line per seed ({len(runs)})", xlabel="epoch")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(ep, np.mean([h["train_loss"] for h in hist], 0), label="train loss (mean)")
    ax[1].plot(ep, np.mean([h["val_loss"] for h in hist], 0), label="val loss (mean)")
    ax[1].set(title="Loss, mean over seeds", xlabel="epoch")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    ax[2].imshow(rows, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax[2].text(j, i, f"{cm[i, j]}\n{rows[i, j]:.0%}", ha="center", va="center",
                       color="white" if rows[i, j] > 0.5 else "black")
    ax[2].set_xticks(range(4))
    ax[2].set_yticks(range(4))
    ax[2].set_xticklabels(CLASSES, rotation=30, ha="right")
    ax[2].set_yticklabels(CLASSES)
    ax[2].set(title="Test confusion matrix, summed over seeds\n(% = share of each true class)",
              xlabel="predicted", ylabel="true")

    fig.tight_layout()
    out_dir = os.path.join(out_root, "_summary")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"arm_{arm}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"arm {arm}: {len(runs)} runs -> {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", default="outputs", help="folder holding the run folders")
    p.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=sorted(ARMS))
    p.add_argument("--n_seeds", type=int, default=5)
    a = p.parse_args()
    for arm in a.arms:
        arm_figure(arm, a.out_root, a.n_seeds)


if __name__ == "__main__":
    main()
