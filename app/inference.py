"""
Model loading and image preprocessing for the Streamlit demo.

No Streamlit imports here, so this module can be tested on its own.

Checkpoints are the best.pt files written by train_transfer.py (best_model.pt for the SimpleCNN
baseline), selected by validation macro-F1. Put them under app/models/ in the same layout the
training runs use:

    app/models/tl_pre_224_s0/best.pt          (arm B, seed 0)
    app/models/tl_pre_224_s1/best.pt
    ...
or as flat files named after the run:

    app/models/tl_pre_224_s0.pt

When an arm has several seeds, the demo averages their softmax probabilities.

Preprocessing reproduces how the training images were prepared: convert to RGB, shrink so the
longest side is at most cache_size (PIL thumbnail), JPEG-encode at the cache quality, then resize
to img_size x img_size and normalise with ImageNet mean/std. This matters because the networks
were trained on those resized JPEG copies, not on full-size photos.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ImageFolder sorts class folders alphabetically; this is the index order used in training.
CLASSES = ["Bad fruit", "Bad leaf", "Good fruit", "Good leaf"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAX_SEEDS = 10


@dataclass(frozen=True)
class ArmSpec:
    key: str
    title: str
    kind: str            # "resnet18" or "simplecnn"
    prefix: str          # run folders are <prefix>_s0, <prefix>_s1, ...
    ckpt: str            # checkpoint file name inside each run folder
    img_size: int        # network input size
    cache_size: int      # longest side of the resized training copy
    cache_quality: int   # JPEG quality of the resized training copy
    test_f1: str         # test macro-F1, mean +- SD over 5 seeds (full test set, n=677)


ARMS: dict[str, ArmSpec] = {s.key: s for s in [
    ArmSpec("B", "Pretrained ResNet18, 224 px, lr 1e-4", "resnet18", "tl_pre_224", "best.pt", 224, 256, 95, "0.884 ± 0.013"),
    ArmSpec("A", "Pretrained ResNet18, 128 px, lr 1e-4", "resnet18", "tl_pre_128", "best.pt", 128, 160, 92, "0.864 ± 0.003"),
    ArmSpec("D", "Pretrained ResNet18, 128 px, lr 1e-3", "resnet18", "tl_pre_128_lr1e3", "best.pt", 128, 160, 92, "0.825 ± 0.013"),
    ArmSpec("base", "SimpleCNN from scratch, 128 px, lr 1e-3", "simplecnn", "run_plain", "best_model.pt", 128, 160, 92, "0.799 ± 0.013"),
    ArmSpec("E", "ResNet18 from scratch, 128 px, lr 1e-4", "resnet18", "tl_scratch_128_lr1e4", "best.pt", 128, 160, 92, "0.792 ± 0.008"),
    ArmSpec("C", "ResNet18 from scratch, 128 px, lr 1e-3", "resnet18", "tl_scratch_128", "best.pt", 128, 160, 92, "0.776 ± 0.039"),
]}


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class SimpleCNN(nn.Module):
    """The baseline from train_baseline.py with batchnorm and dropout off (the 'plain' run).

    Layer indices match that script, so its state_dict loads unchanged
    (features.<i>.0.weight ... classifier.2.weight).
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128), _block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Identity(), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(kind: str, num_classes: int = 4) -> nn.Module:
    if kind == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if kind == "simplecnn":
        return SimpleCNN(num_classes)
    raise ValueError(f"unknown model kind: {kind}")


def find_checkpoints(models_dir: Path, spec: ArmSpec) -> list[tuple[str, Path]]:
    """(run name, path) for every seed of this arm found under models_dir."""
    found = []
    for seed in range(MAX_SEEDS):
        run = f"{spec.prefix}_s{seed}"
        for candidate in (models_dir / run / spec.ckpt, models_dir / f"{run}.pt"):
            if candidate.is_file():
                found.append((run, candidate))
                break
    return found


def available_arms(models_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """Arms that have at least one checkpoint, in display order."""
    out = {}
    for key, spec in ARMS.items():
        ckpts = find_checkpoints(Path(models_dir), spec)
        if ckpts:
            out[key] = ckpts
    return out


@dataclass
class LoadedArm:
    spec: ArmSpec
    names: list[str]
    nets: list[nn.Module]


def load_arm(key: str, models_dir: Path) -> LoadedArm:
    spec = ARMS[key]
    ckpts = find_checkpoints(Path(models_dir), spec)
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints for arm {key} under {models_dir} "
                                f"(expected {spec.prefix}_s0/{spec.ckpt} or {spec.prefix}_s0.pt)")
    names, nets = [], []
    for run, path in ckpts:
        net = build_model(spec.kind)
        try:
            # weights_only: a checkpoint is just tensors, so refuse anything that would run code
            state = torch.load(path, map_location="cpu", weights_only=True)
            net.load_state_dict(state)
        except Exception as e:  # wrong architecture, corrupt file, ...
            raise RuntimeError(f"could not load {path.name} for arm {key} ({spec.kind}): {e}") from e
        names.append(run)
        nets.append(net.eval())
    return LoadedArm(spec, names, nets)


def to_rgb(image: Image.Image) -> Image.Image:
    """RGB copy; transparent pixels go onto white (the dataset's cut-outs sit on white)."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def preprocess(image: Image.Image, spec: ArmSpec) -> torch.Tensor:
    """Same steps as the resized training copies, then the training transform. Returns (1, 3, H, W)."""
    im = to_rgb(image)
    im.thumbnail((spec.cache_size, spec.cache_size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=spec.cache_quality)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((spec.img_size, spec.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tfm(im).unsqueeze(0)


def predict_arm(arm: LoadedArm, image: Image.Image) -> np.ndarray:
    """Softmax probabilities, shape (n_checkpoints, 4)."""
    x = preprocess(image, arm.spec)
    with torch.inference_mode():
        return np.stack([torch.softmax(net(x), dim=1)[0].numpy() for net in arm.nets])
