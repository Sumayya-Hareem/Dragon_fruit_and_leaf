"""
Baseline CNN from scratch — Dragon fruit & leaf classifier (4 classes).

Run 1 is deliberately plain: no batch norm, no dropout, no class weighting.
Each later experiment flips ONE switch so you can see what that technique changes:

    python train_baseline.py --data_root /content/cleaned --run_name run1_plain
    python train_baseline.py --data_root /content/cleaned --run_name run2_bn --batchnorm
    python train_baseline.py --data_root /content/cleaned --run_name run3_bn_dropout --batchnorm --dropout 0.3
    python train_baseline.py --data_root /content/cleaned --run_name run4_weighted --batchnorm --dropout 0.3 --class_weights

Outputs (in outputs/<run_name>/):
    best_model.pt          weights from the epoch with the best validation macro-F1
    history.json           per-epoch loss/accuracy/F1 (for comparing runs later)
    curves.png             train vs val loss and accuracy — your main diagnostic
    confusion_matrix.png   test-set confusion matrix
    test_report.txt        per-class precision / recall / F1 on the test set
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots without needing a display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------- arguments
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="folder containing train/ val/ test/")
    p.add_argument("--run_name", type=str, default="run1_plain")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batchnorm", action="store_true", help="add BatchNorm after each conv")
    p.add_argument("--dropout", type=float, default=0.0, help="dropout before the final layer (0 = off)")
    p.add_argument("--class_weights", action="store_true", help="weight the loss by inverse class frequency")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    # Same seed = same weight init and data order, so runs are comparable.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
def build_loaders(data_root, img_size, batch_size):
    # Normalizing with ImageNet stats is a common convention; it just centers
    # pixel values around 0 so gradients behave better early in training.
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # No augmentation in the baseline on purpose — that's a later experiment.
    root = Path(data_root)
    train_ds = datasets.ImageFolder(root / "train", transform=tfm)
    val_ds = datasets.ImageFolder(root / "val", transform=tfm)
    test_ds = datasets.ImageFolder(root / "test", transform=tfm)

    assert train_ds.classes == val_ds.classes == test_ds.classes, "class folders differ between splits"

    kw = dict(batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        train_ds,
    )


# ---------------------------------------------------------------- model
def conv_block(c_in, c_out, batchnorm):
    layers = [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(c_out))
    layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    """
    4 conv blocks. Each block: conv (learns local patterns) -> ReLU -> max-pool (halves size).
    128px input -> 64 -> 32 -> 16 -> 8. Channels grow 32 -> 64 -> 128 -> 256 so
    the network trades spatial detail for more, richer features as it goes deeper.
    """
    def __init__(self, num_classes, batchnorm=False, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, batchnorm),
            conv_block(32, 64, batchnorm),
            conv_block(64, 128, batchnorm),
            conv_block(128, 256, batchnorm),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # average each feature map to one number
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# ---------------------------------------------------------------- train / eval
def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise just evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()   # clear old gradients
                loss.backward()         # backprop: compute d(loss)/d(every weight)
                optimizer.step()        # nudge weights against the gradient

            total_loss += loss.item() * images.size(0)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        "loss": total_loss / len(labels),
        "acc": float((preds == labels).mean()),
        # Macro-F1 averages F1 across classes equally, so a model that ignores
        # the small classes gets punished even if overall accuracy looks fine.
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "preds": preds,
        "labels": labels,
    }


def save_curves(history, out_dir):
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train acc")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val acc")
    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], label="val macro-F1", linestyle="--")
    axes[1].set_title("Accuracy / macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "curves.png", dpi=120)
    plt.close()


def save_confusion(labels, preds, class_names, out_dir):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU found — this will be very slow. Run on Colab with a T4 runtime.")

    out_dir = Path("outputs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader, test_loader, train_ds = build_loaders(
        args.data_root, args.img_size, args.batch_size)
    class_names = train_ds.classes
    print(f"Classes: {class_names}")

    model = SimpleCNN(len(class_names), args.batchnorm, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    weight = None
    if args.class_weights:
        counts = np.bincount(train_ds.targets, minlength=len(class_names))
        w = counts.sum() / (len(class_names) * counts)  # rare class -> bigger weight
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        print("Class weights:", dict(zip(class_names, np.round(w, 2))))
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": tr["loss"], "train_acc": tr["acc"],
            "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": va["macro_f1"],
        })
        flag = ""
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            flag = "  <- best"
        print(f"Epoch {epoch:02d} | train loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
              f"val loss {va['loss']:.3f} acc {va['acc']:.3f} F1 {va['macro_f1']:.3f} | "
              f"{time.time() - t0:.0f}s{flag}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    save_curves(history, out_dir)

    # Test ONCE, with the best checkpoint. Never tune anything on test results.
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    te = run_epoch(model, test_loader, criterion, device)
    report = classification_report(te["labels"], te["preds"], target_names=class_names, digits=3)
    print("\n=== Test set ===\n" + report)
    with open(out_dir / "test_report.txt", "w") as f:
        f.write(report)
    save_confusion(te["labels"], te["preds"], class_names, out_dir)
    print(f"Saved everything to {out_dir}")


if __name__ == "__main__":
    main()
