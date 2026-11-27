"""
find_near_duplicates.py -- near-duplicate and label-conflict audit for an ImageFolder dataset.

Question it answers: does an image in val/test have a near-identical "twin" in train,
and if so, does the twin carry the same label?

Method
  * Embed every image with an ImageNet-pretrained ResNet18 (fc = Identity, 512-d),
    ImageNet normalisation, resized to --img_size (default 224).
  * L2-normalise, so a dot product is the cosine similarity.
  * For every val/test image, find its nearest train image (cosine nearest neighbour).
    A "twin" is a pair with cosine >= --tau (default 0.98).
  * A "label conflict" is a twin whose nearest train image has a DIFFERENT label.
  * Also reports how often train images have a twin elsewhere in train (the baseline rate).

A perceptual hash (dHash) was tried first and saturated at 76-80% in every row, so it is
not used here; the embedding separates same-photo from merely-similar-leaf much better.

Outputs (in --out_dir, default outputs/_summary):
  nearest_neighbours_<split>.csv   every image in the split: nearest train image, cosine,
                                   labels, is_twin, label_conflict
  label_conflict_pairs.csv         the twins whose labels differ
  label_conflict_pairs.png         gallery of the most similar conflicting pairs
  near_dup_summary.json            the headline numbers, for provenance

The nearest_neighbours CSV is also what a "twin-free" evaluation needs: keep the test
images whose cosine is < tau (rows are in ImageFolder order, same as the training script).

Note on img_size: the first analysis ran on 128 px images and found 181 test twins
(20 label conflicts); re-running at 224 px finds 182 (21). Use --img_size 128 to
reproduce the first one.

Usage
  python find_near_duplicates.py --data_root Cleaned
  python find_near_duplicates.py --data_root Cleaned --split val --tau 0.95
"""
import argparse
import csv
import json
import os
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, help="folder containing train/ val/ test/ (ImageFolder layout)")
    p.add_argument("--out_dir", default=os.path.join("outputs", "_summary"))
    p.add_argument("--split", choices=["val", "test"], default="test",
                   help="split whose twins and label conflicts are listed (default: test)")
    p.add_argument("--tau", type=float, default=0.98, help="cosine cutoff for 'has a twin in train'")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--n_pairs", type=int, default=12, help="pairs shown in the gallery (even number)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--weights", choices=["imagenet", "none"], default="imagenet",
                   help="'none' = random features, for a smoke test only")
    return p.parse_args()


def load_split(root, name, tfm):
    for n in (["val", "valid"] if name == "val" else [name]):
        d = os.path.join(root, n)
        if os.path.isdir(d):
            return datasets.ImageFolder(d, tfm)
    raise FileNotFoundError(f"no '{name}' folder in {root}. Folders present: {sorted(os.listdir(root))}")


@torch.no_grad()
def embed(net, ds, device, batch_size, num_workers):
    out = []
    for x, _ in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers):
        f = net(x.to(device)).cpu().numpy()
        out.append(f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12))
    return np.concatenate(out)


def name_style(path):
    """Leading letters of the file name, upper-cased: 'PXL_2023..._1.jpg' -> 'PXL', 'BGR (12).jpg' -> 'BGR'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"[A-Za-z]+", stem)
    return m.group(0).upper() if m else "OTHER"


def pair_style(a, b):
    return "+".join(sorted([name_style(a), name_style(b)]))


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    tfm = transforms.Compose([
        transforms.Resize((a.img_size, a.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = {s: load_split(a.data_root, s, tfm) for s in ["train", "val", "test"]}
    classes = ds["train"].classes
    assert ds["val"].classes == classes == ds["test"].classes, "class folders differ between splits"
    paths = {s: [p for p, _ in ds[s].samples] for s in ds}
    labels = {s: np.array(ds[s].targets) for s in ds}
    print(f"Classes: {classes}")
    print("Images: " + "  ".join(f"{s} {len(paths[s])}" for s in ds), flush=True)

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if a.weights == "imagenet" else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Identity()
    net = net.eval().to(device)

    emb = {}
    for s in ds:
        emb[s] = embed(net, ds[s], device, a.batch_size, a.num_workers)
        print(f"embedded {s}: {emb[s].shape}", flush=True)

    # ---- nearest train neighbour for every image ----
    nn_sim, nn_idx = {}, {}
    for s in ["val", "test"]:
        S = emb[s] @ emb["train"].T
        nn_sim[s], nn_idx[s] = S.max(1), S.argmax(1)
    S = emb["train"] @ emb["train"].T
    np.fill_diagonal(S, -1.0)                       # nearest OTHER train image
    nn_sim["train"], nn_idx["train"] = S.max(1), S.argmax(1)
    del S

    print("\nShare of images with a near-identical twin in train (cosine similarity):")
    print(f"{'':24s}{'n':>6s}{'median':>9s}{'>=0.95':>9s}{'>=0.98':>9s}")
    shares = {}
    for s, name in [("train", "train -> other train"), ("val", "val -> train"), ("test", "test -> train")]:
        x = nn_sim[s]
        shares[name] = {"n": int(len(x)), "median": float(np.median(x)),
                        "share_ge_0.95": float((x >= 0.95).mean()), "share_ge_0.98": float((x >= 0.98).mean())}
        print(f"{name:24s}{len(x):>6d}{np.median(x):>9.3f}{(x >= .95).mean():>9.1%}{(x >= .98).mean():>9.1%}")

    vt_sim = np.concatenate([nn_sim["val"], nn_sim["test"]])
    vt_same = np.concatenate([labels[s] == labels["train"][nn_idx[s]] for s in ["val", "test"]])
    n95 = int((vt_sim >= 0.95).sum())
    same95 = float(vt_same[vt_sim >= 0.95].mean()) if n95 else float("nan")
    print(f"\nval+test images with a >=0.95 train neighbour: {n95}; same label in {same95:.0%} of them")

    # ---- twins and label conflicts for the chosen split ----
    s = a.split
    y = labels[s]
    sim, idx = nn_sim[s], nn_idx[s]
    twin_label = labels["train"][idx]
    twin = sim >= a.tau
    same = twin & (twin_label == y)
    conflict = twin & (twin_label != y)
    print(f"\n{s} images with a train twin (cosine >= {a.tau}): {twin.sum()} of {len(y)} "
          f"-> same label {same.sum()}, different label {conflict.sum()}"
          + (f" ({conflict.sum() / twin.sum():.1%} of twins)" if twin.sum() else ""))
    composition = Counter(f"{classes[y[i]]} -> {classes[twin_label[i]]}" for i in np.where(conflict)[0])
    print(f"label conflicts ({s} label -> twin label): {dict(composition)}")

    # file-name pattern: do conflicting pairs come from two different naming schemes?
    style_same = Counter(pair_style(paths[s][i], paths["train"][idx[i]]) for i in np.where(same)[0])
    style_conf = Counter(pair_style(paths[s][i], paths["train"][idx[i]]) for i in np.where(conflict)[0])
    print(f"file-name styles of same-label twin pairs:  {dict(style_same)}")
    print(f"file-name styles of conflicting pairs:      {dict(style_conf)}")

    # ---- write outputs ----
    os.makedirs(a.out_dir, exist_ok=True)
    nn_csv = os.path.join(a.out_dir, f"nearest_neighbours_{s}.csv")
    conflict_csv = os.path.join(a.out_dir, "label_conflict_pairs.csv")
    head = ["split", "query_path", "train_path", "cosine", "query_label", "train_label", "is_twin", "label_conflict"]
    with open(nn_csv, "w", newline="") as f_all, open(conflict_csv, "w", newline="") as f_conf:
        w_all, w_conf = csv.writer(f_all), csv.writer(f_conf)
        w_all.writerow(head)
        w_conf.writerow(head)
        for i in range(len(y)):
            row = [s, paths[s][i], paths["train"][idx[i]], f"{sim[i]:.4f}",
                   classes[y[i]], classes[twin_label[i]], int(twin[i]), int(conflict[i])]
            w_all.writerow(row)
            if conflict[i]:
                w_conf.writerow(row)
    print(f"\nwrote {nn_csv}\nwrote {conflict_csv}")

    summary = {
        "data_root": a.data_root, "split": s, "tau": a.tau, "img_size": a.img_size, "weights": a.weights,
        "n_images": int(len(y)), "twins": int(twin.sum()), "twins_same_label": int(same.sum()),
        "twins_label_conflict": int(conflict.sum()),
        "label_conflict_composition": dict(composition),
        "filename_styles_same_label": dict(style_same), "filename_styles_conflict": dict(style_conf),
        "val_test_ge_0.95_neighbours": n95, "val_test_ge_0.95_same_label_share": same95,
        "shares": shares,
    }
    with open(os.path.join(a.out_dir, "near_dup_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- gallery: most similar conflicting pairs ----
    order = np.where(conflict)[0]
    order = order[np.argsort(-sim[order])][: a.n_pairs]
    if len(order):
        rows = (len(order) + 1) // 2
        fig, ax = plt.subplots(rows, 4, figsize=(13, 3.6 * rows), squeeze=False)
        for axis in ax.ravel():
            axis.axis("off")
        for k, i in enumerate(order):
            r, c = divmod(k, 2)
            j = idx[i]
            panels = [
                (paths[s][i], f"{s}: {classes[y[i]]}\n{os.path.basename(paths[s][i])}"),
                (paths["train"][j], f"train: {classes[twin_label[i]]}  (cos {sim[i]:.3f})\n"
                                    f"{os.path.basename(paths['train'][j])}"),
            ]
            for m, (p, title) in enumerate(panels):
                ax[r, 2 * c + m].imshow(Image.open(p).convert("RGB"))
                ax[r, 2 * c + m].set_title(title, fontsize=7)
        fig.suptitle(f"{s} images whose near-identical train twin has a DIFFERENT label", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(a.out_dir, "label_conflict_pairs.png"), dpi=110)
        plt.close(fig)
        print(f"wrote {os.path.join(a.out_dir, 'label_conflict_pairs.png')}")
    else:
        print("no label conflicts at this cutoff, so no gallery was drawn")


if __name__ == "__main__":
    main()
