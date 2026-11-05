"""Training loop for the fusion U-Net.

CPU-friendly: a few hundred steps on procedural data is enough to beat the
classical score on the synthetic distribution. Real deployments should fine
tune on camera-native tampered pairs (see README).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import TamperDataset
from .unet import UNet, count_params


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def batch_iou(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum().item()
    union = ((pred + target) > 0).sum().item()
    return inter / max(union, 1.0)


def train(out: str | Path = "edittrace/weights/edittrace_unet.pt",
          steps: int = 420, batch: int = 4, size: int = 256,
          lr: float = 2.2e-4, data: str | None = None,
          width: int = 24, seed: int = 0, log_every: int = 40,
          val_every: int = 70, patience: int = 8) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TamperDataset(length=steps * batch + 10, size=(size, size),
                             n_edits=(1, 2), seed_offset=1_000, folder=data)
    val_seeds = [90_001, 90_002, 90_003, 90_004, 90_005, 90_006]
    val_ds = TamperDataset(length=len(val_seeds), size=(size, size),
                           n_edits=(1, 2), seed_offset=0, folder=data)

    model = UNet(in_ch=18, width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.12, anneal_strategy="cos")

    print(f"device={device} params={count_params(model):,} steps={steps} "
          f"batch={batch} size={size} data={data or 'procedural'}")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    bad = 0
    hist = []
    t0 = time.time()

    # fixed validation tensors
    val_batches = []
    for i in range(len(val_seeds)):
        vx, vy = val_ds[i]
        val_batches.append((vx[None].to(device), vy[None].to(device)))

    for step in range(1, steps + 1):
        idxs = np.random.default_rng(seed * 10_000 + step).integers(
            0, len(train_ds), size=batch)
        xs, ys = [], []
        for i in idxs:
            x, y = train_ds[int(i)]
            xs.append(x)
            ys.append(y)
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)

        model.train()
        logits = model(x)
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(2.2))
        loss = bce + dice_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                iou = batch_iou(logits, y)
            print(f"step {step:4d}/{steps}  loss {loss.item():.4f}  "
                  f"train_iou {iou:.3f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{time.time() - t0:.1f}s")
            hist.append({"step": step, "loss": float(loss.item()), "train_iou": float(iou)})

        if step % val_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vloss, viou = 0.0, 0.0
                for vx, vy in val_batches:
                    vl = model(vx)
                    vloss += float(F.binary_cross_entropy_with_logits(vl, vy))
                    viou += batch_iou(vl, vy)
                vloss /= len(val_batches)
                viou /= len(val_batches)
            print(f"  val loss {vloss:.4f}  val_iou {viou:.3f}")
            hist.append({"step": step, "val_loss": vloss, "val_iou": viou})
            metric = viou - 0.05 * vloss
            if metric > best:
                best = metric
                bad = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "width": width,
                    "depth": 4,
                    "channels": 13,
                    "metrics": {"val_iou": viou, "val_loss": vloss, "step": step},
                    "train": {"steps": steps, "batch": batch, "size": size,
                              "data": data or "procedural"},
                }, out_path)
                print(f"  saved -> {out_path}")
            else:
                bad += 1
                if bad >= patience:
                    print("early stop")
                    break

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps({
        "steps": steps, "batch": batch, "size": size, "lr": lr,
        "params": count_params(model), "history": hist,
        "elapsed_s": round(time.time() - t0, 2),
    }, indent=2), encoding="utf-8")
    print(f"done in {time.time() - t0:.1f}s -> {out_path}")
    return {"checkpoint": str(out_path), "history": hist}


if __name__ == "__main__":  # pragma: no cover
    train()
