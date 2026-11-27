"""
train_transfer.py -- transfer-learning counterpart to train_baseline.py

Same conventions as the baseline so results are comparable:
  * Adam, batch 32, 25 epochs, seed set by --seed
  * checkpoint selected by VALIDATION MACRO-F1
  * test set evaluated exactly ONCE, with the best checkpoint
  * everything written under outputs/<run_name>/ as it happens
    (history.json after every epoch, so a Colab disconnect loses nothing)

What differs from the baseline (and why it has to):
  * model: torchvision ResNet18 with a fresh 4-class head, either
      --weights imagenet   pretrained   (the transfer-learning arm)
      --weights none       random init  (the architecture control)
  * input normalisation: ImageNet mean/std. A pretrained network's first layer
    was trained on inputs with that scale; feeding it anything else wastes part
    of what it learned. Applied to BOTH arms so weights are the only difference.
  * default lr 1e-4 instead of 1e-3: fine-tuning moves weights that are already
    useful, so it takes smaller steps. Override with --lr (e.g. for the
    random-init control).

No augmentation, matching the baseline.

Usage (Colab):
  python train_transfer.py --data_root /content/small --run_name tl_pre_128_s0 --seed 0
"""
import argparse
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="folder containing train/ val/ test/ (ImageFolder layout)")
    p.add_argument("--run_name", required=True)
    p.add_argument("--out_root", default="outputs")
    p.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--allow_cpu", action="store_true",
                   help="by default the script refuses to train without a GPU")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def find_split(root, names):
    for n in names:
        d = os.path.join(root, n)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(
        f"none of {names} found in {root}. Folders present: {sorted(os.listdir(root))}")


def build_model(weights, n_classes):
    w = models.ResNet18_Weights.IMAGENET1K_V1 if weights == "imagenet" else None
    m = models.resnet18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, n_classes)  # fresh head either way
    return m


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()  # BatchNorm uses running statistics here, not batch statistics
    total_loss, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        preds.append(out.argmax(1).cpu())
        labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    loss = total_loss / len(labels)
    acc = float((preds == labels).mean())
    f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return loss, acc, f1, preds, labels


def save_curves(h, path):
    ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, h["train_loss"], label="train")
    ax[0].plot(ep, h["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(ep, h["train_acc"], label="train acc")
    ax[1].plot(ep, h["val_acc"], label="val acc")
    ax[1].plot(ep, h["val_f1"], label="val macro-F1")
    ax[1].set_xlabel("epoch"); ax[1].set_title("Accuracy / macro-F1"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def save_confusion(cm, names, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Test confusion matrix")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    a = parse_args()
    seed_everything(a.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cpu" and not a.allow_cpu:
        raise SystemExit("No GPU found. Check the Colab runtime type (T4) and re-run "
                         "setup; pass --allow_cpu only for a smoke test.")

    tfm = transforms.Compose([
        transforms.Resize((a.img_size, a.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = datasets.ImageFolder(find_split(a.data_root, ["train"]), tfm)
    val_ds = datasets.ImageFolder(find_split(a.data_root, ["val", "valid"]), tfm)
    test_ds = datasets.ImageFolder(find_split(a.data_root, ["test"]), tfm)
    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx, \
        "class folders differ between splits"
    class_names = train_ds.classes
    print(f"Classes: {class_names}")
    print(f"Images: train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}", flush=True)

    g = torch.Generator()
    g.manual_seed(a.seed)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                              num_workers=a.num_workers, pin_memory=pin,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=a.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=a.num_workers, pin_memory=pin)

    model = build_model(a.weights, len(class_names)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet18, weights={a.weights}, params={n_params:,}, "
          f"img_size={a.img_size}, lr={a.lr}, seed={a.seed}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)

    out_dir = os.path.join(a.out_root, a.run_name)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.pt")

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1",
                            "epoch_seconds"]}
    best_f1, best_epoch = -1.0, 0

    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, run_correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            run_correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
        tr_loss, tr_acc = run_loss / n, run_correct / n

        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, criterion, device)
        secs = time.time() - t0

        for k, v in zip(hist.keys(), [tr_loss, tr_acc, va_loss, va_acc, va_f1, secs]):
            hist[k].append(v)
        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch
            torch.save(model.state_dict(), best_path)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(hist, f)

        print(f"epoch {epoch:02d}/{a.epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} F1 {va_f1:.3f} | "
              f"{secs:.1f}s{'  *best' if best_epoch == epoch else ''}", flush=True)

    # ---- test: once, best checkpoint only ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc, te_f1, preds, labels = evaluate(model, test_loader, criterion, device)
    report = classification_report(labels, preds, target_names=class_names, digits=3,
                                   zero_division=0)
    cm = confusion_matrix(labels, preds)
    per_class_recall = (cm.diagonal() / cm.sum(1)).tolist()

    header = (f"run: {a.run_name}\nbest epoch (val macro-F1): {best_epoch}  "
              f"val macro-F1 {best_f1:.4f}\n"
              f"TEST accuracy {te_acc:.4f}  macro-F1 {te_f1:.4f}\n\n")
    with open(os.path.join(out_dir, "test_report.txt"), "w") as f:
        f.write(header + report + "\nconfusion matrix (rows true, cols predicted):\n"
                + str(cm) + "\n")
    save_curves(hist, os.path.join(out_dir, "curves.png"))
    save_confusion(cm, class_names, os.path.join(out_dir, "confusion_matrix.png"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({
            "run_name": a.run_name, "weights": a.weights, "img_size": a.img_size,
            "lr": a.lr, "seed": a.seed, "epochs": a.epochs, "params": n_params,
            "best_epoch": best_epoch, "val_macro_f1": best_f1,
            "test_acc": te_acc, "test_macro_f1": te_f1,
            "per_class_recall": dict(zip(class_names, per_class_recall)),
            "mean_epoch_seconds": float(np.mean(hist["epoch_seconds"])),
        }, f, indent=2)

    print("\n" + header + report, flush=True)


if __name__ == "__main__":
    main()
