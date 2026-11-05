# Checkpoints

- `edittrace_unet.pt` — fusion U-Net, cold-started on procedural scenes.
- `edittrace_real.pt` — same architecture fine-tuned on real spliced photographs
  (Columbia train split), for camera-native inputs.
- `precedence.npz` — learned pairwise ordering model.

## `edittrace_unet.pt`

- Architecture: `UNet(in_ch=18, width=24, depth=4)`, 1.1M parameters
- Trained on: procedurally tampered scenes (splice / clone / inpaint, multi-session
  JPEG saves, feathered blends, resampling)
- Input: 18 channels — 12 forensic maps, luma, local luma mean/std, three contrast maps
- Output: single-channel manipulation logit at input resolution

The companion `.json` file records the training history and step count.

## `edittrace_real.pt`

Fine-tuned from the synthetic checkpoint on 215 real photographs from the Columbia
splicing benchmark (106 forged with masks, 109 authentic), balanced batches with crops
centred on the tampered region, 200 steps at 320 px, JPEG augmentation. On the held-out
Columbia test split this improves pixel IoU from 0.118 to 0.296 and image-level MCC from
0.000 to 0.441. Use it for camera-native photos:

```bash
python -m edittrace analyze photo.jpg --model edittrace/weights/edittrace_real.pt
```

## `precedence.npz`

Antisymmetric logistic model over 18 pair features, trained on 160 synthetic scenes with
known edit order. Blended 78/22 with the rule-based precedence score.

## Reproduce

```bash
python -m edittrace train --steps 400 --batch 6 --size 192
python -m edittrace train-order --scenes 160
python scripts/finetune_real.py --dataset columbia --steps 200 --size 320
```

