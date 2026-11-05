"""Fine-tune the fusion U-Net on real tampered pairs.

Example (COVERAGE):
    python scripts/finetune_real.py --root data/real --steps 160 --size 320
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edittrace.data import columbia_rows, coverage_rows
from edittrace.models.real_dataset import PairedMaskDataset
from edittrace.models.train import batch_iou, dice_loss
from edittrace.models.unet import UNet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/real")
    ap.add_argument("--dataset", default="columbia", choices=["coverage", "columbia"])
    ap.add_argument("--init", default="edittrace/weights/edittrace_unet.pt")
    ap.add_argument("--out", default="edittrace/weights/edittrace_real.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = columbia_rows if a.dataset == "columbia" else coverage_rows
    rows = loader(a.root, split="train")
    val = [r for i, r in enumerate(rows) if i % 9 == 0]
    train = [r for i, r in enumerate(rows) if i % 9 != 0]
    forged = [i for i, r in enumerate(train) if r["tampered"]]
    authentic = [i for i, r in enumerate(train) if not r["tampered"]]
    print(f"{a.dataset}: train {len(train)} ({len(forged)} forged), val {len(val)}")

    ds = PairedMaskDataset(train, size=(a.size, a.size), train=True)
    vds = PairedMaskDataset(val, size=(a.size, a.size), train=False, jpeg_aug=0.0)

    ckpt = torch.load(a.init, map_location="cpu", weights_only=False)
    width = int(ckpt.get("width", 24))
    model = UNet(in_ch=18, width=width).to(device)
    model.load_state_dict(ckpt["state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.2, anneal_strategy="cos")

    best = -1.0
    t0 = time.time()
    for step in range(1, a.steps + 1):
        rng = np.random.default_rng(step)
        # balanced batch: half forged, half authentic
        n_forged = a.batch // 2
        idxs = list(rng.choice(forged, n_forged, replace=len(forged) < n_forged))
        idxs += list(rng.choice(authentic, a.batch - n_forged,
                                replace=len(authentic) < (a.batch - n_forged)))
        xs, ys = zip(*[ds[int(i)] for i in idxs])
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        model.train()
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 20 == 0 or step == 1:
            print(f"step {step:4d}/{a.steps} loss {loss.item():.4f} "
                  f"iou {batch_iou(logits, y):.3f} {time.time() - t0:.0f}s")
        if step % 40 == 0 or step == a.steps:
            model.eval()
            ious, fps = [], []
            with torch.no_grad():
                for i, row in enumerate(val):
                    vx, vy = vds[i]
                    vl = model(vx[None].to(device))
                    pred = (torch.sigmoid(vl) > 0.5).float()
                    if row["tampered"]:
                        inter = float((pred * vy[None]).sum())
                        union = float(((pred + vy[None]) > 0).sum())
                        ious.append(inter / max(union, 1.0))
                    else:
                        fps.append(float(pred.mean()))
            viou = float(np.mean(ious)) if ious else 0.0
            vfp = float(np.mean(fps)) if fps else 0.0
            metric = viou - 0.5 * vfp
            print(f"  val forged IoU {viou:.3f}  authentic FP area {vfp:.3f}")
            if metric > best:
                best = metric
                torch.save({"state_dict": model.state_dict(), "width": width,
                            "depth": 4, "channels": 18,
                            "metrics": {"val_iou": viou, "val_fp_area": vfp,
                                        "step": step, "dataset": a.dataset}},
                           a.out)
                print(f"  saved -> {a.out}")

    meta = {"dataset": a.dataset, "steps": a.steps, "batch": a.batch,
            "size": a.size, "lr": a.lr, "best_val_iou": best,
            "elapsed_s": round(time.time() - t0, 1), "init": a.init,
            "train_rows": len(train), "val_rows": len(val)}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
