"""Train the mini GPT:  python -m llm.train --steps 3000

  --holdout 0.15   keep 15% of areas unseen in training and report how well the model copes with them
  --holdout 0      train on every row (this is the model the app uses)
"""
import argparse
import math
import random
import re
import time
from pathlib import Path

import numpy as np

from . import tinygpt as G
from .corpus import ROOT, is_holdout, load_rows, make_example


def batch(rows, pool, rng, B, block):
    ex = [make_example(rows, rng, pool) for _ in range(B)]
    seqs = [(G.encode(p), G.encode(a + "\n")) for p, a in ex]
    T = min(block + 1, max(len(p) + len(a) for p, a in seqs))
    idx = np.zeros((B, T - 1), dtype=np.int64)
    tgt = np.zeros((B, T - 1), dtype=np.int64)
    msk = np.zeros((B, T - 1), dtype=np.float32)
    for i, (p, a) in enumerate(seqs):
        s = (p + a)[:T]
        idx[i, :len(s) - 1] = s[:-1]
        tgt[i, :len(s) - 1] = s[1:]
        msk[i, len(p) - 1:len(s) - 1] = 1.0                  # learn only the answer, not the facts or question
    return idx, tgt, msk


def evaluate(p, cfg, rows, test_rows, n=60, seed=1):
    """Greedy-decode held-out examples. Reports exact match and 'numbers copied correctly'."""
    rng = random.Random(seed)
    exact = nums_ok = 0
    for _ in range(n):
        prompt, want = make_example(rows, rng, test_rows, synth=0.0)
        got = G.generate(p, cfg, prompt, max_new=260)
        exact += got.strip() == want.strip()
        facts = set(re.findall(r"\d[\d,]*\.?\d*", prompt))
        nums_ok += all(x.rstrip(".,") in {f.rstrip(".,") for f in facts} for x in re.findall(r"\d[\d,]*\.?\d*", got))
    return exact / n, nums_ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--holdout", type=float, default=0.0)
    ap.add_argument("--out", default=str(ROOT / "llm" / "model.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="continue training from a saved model")
    a = ap.parse_args()
    rows = load_rows()
    held = [r for r in rows if is_holdout(r["area"], a.holdout)]
    pool = [r for r in rows if not is_holdout(r["area"], a.holdout)]
    if a.init:
        p, cfg = G.load(a.init)
    else:
        cfg = G.Config(block=288, d=a.d, layers=a.layers, heads=4)
        p = G.init_params(cfg, a.seed)
    opt = G.Adam(p, lr=a.lr)
    rng = random.Random(a.seed)
    print(f"params {G.n_params(p):,}  train rows {len(pool)}  held-out rows {len(held)}", flush=True)
    t0, ema = time.time(), None
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        idx, tgt, msk = batch(rows, pool, rng, a.batch, cfg.block)
        loss, g = G.loss_and_grads(p, cfg, idx, tgt, msk)
        opt.step(p, g, lr)
        ema = loss if ema is None else 0.98 * ema + 0.02 * loss
        if step % 100 == 0 or step == a.steps:
            print(f"step {step:5d}  loss {ema:.3f}  {time.time() - t0:5.0f}s", flush=True)
        if step % 500 == 0:
            G.save(a.out, p, cfg)
    G.save(a.out, p, cfg)
    em, ok = evaluate(p, cfg, rows, held or pool)
    print(f"{'held-out' if held else 'training'} rows: exact match {em:.0%}, numbers all copied from facts {ok:.0%}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
