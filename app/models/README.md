# Model checkpoints for the demo

Weights are not stored in git (`*.pt` is ignored). Copy the checkpoints you want to try from the
training outputs (Google Drive: `dragon_fruit/outputs/`) into this folder, keeping one sub-folder per run:

```
app/models/
├── tl_pre_224_s0/best.pt        arm B (pretrained ResNet18, 224 px), seeds 0-4
├── tl_pre_224_s1/best.pt
├── ...
├── tl_scratch_128_lr1e4_s0/best.pt   arm E (ResNet18 from scratch, 128 px)
└── run_plain_s0/best_model.pt        baseline SimpleCNN (from the older dragonfruit/outputs folder)
```

Flat files named after the run (`tl_pre_224_s0.pt`) work too.

| Arm | Run folder prefix | File | Input |
|---|---|---|---|
| B | `tl_pre_224` | `best.pt` | 224 px |
| A | `tl_pre_128` | `best.pt` | 128 px |
| D | `tl_pre_128_lr1e3` | `best.pt` | 128 px |
| E | `tl_scratch_128_lr1e4` | `best.pt` | 128 px |
| C | `tl_scratch_128` | `best.pt` | 128 px |
| baseline | `run_plain` | `best_model.pt` | 128 px |

If an arm has several seeds, the demo averages their probabilities and shows each seed on request.
Each ResNet18 checkpoint is about 45 MB. To use another folder, set `DRAGON_MODELS_DIR`.
