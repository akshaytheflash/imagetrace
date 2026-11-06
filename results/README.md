# Evaluation artifacts

Machine-readable outputs behind the numbers in the README. Every file includes the
per-image rows so the aggregates can be recomputed.

| file | what it is |
| --- | --- |
| `synthetic_fusion.json` | 12 held-out procedural scenes, fusion model + learned precedence |
| `synthetic_classical.json` | same scenes, classical fallback only |
| `real_columbia_zeroshot.json` | Columbia test split (60 forged / 61 authentic), synthetic checkpoint |
| `real_columbia_finetuned.json` | same split, checkpoint fine-tuned on the Columbia train split |
| `real_coverage_zeroshot.json` | COVERAGE test split (34 / 34), copy-move negative result |

Reproduce:

```bash
python -m edittrace eval --n 12 --out .out
python -m edittrace eval --n 12 --no-model --out .out
python scripts/eval_real.py --dataset columbia --split test --max-side 512 \
  --model edittrace/weights/edittrace_real.pt --model-weight 1.0 --threshold 0.60
python scripts/eval_real.py --dataset coverage --split test --max-side 512
```

Data source: Columbia and COVERAGE archives from the
[ForgeryVCR mirror](https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR)
(`datasets/test/*_tp.zip`, `*_au.zip`). The datasets keep their original licences;
these files contain metrics only, no images.
