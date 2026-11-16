import random, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from llm import tinygpt as G
from llm.corpus import load_rows, make_example, facts_line
from llm.train import batch


def test_every_gradient_matches_finite_differences():
    cfg = G.Config(block=12, d=16, layers=2, heads=2)
    rng = np.random.default_rng(3)
    p = {k: v.astype(np.float64) + rng.normal(0, .05, v.shape) for k, v in G.init_params(cfg, 1).items()}
    idx, tg = rng.integers(1, G.VOCAB, (3, 9)), rng.integers(1, G.VOCAB, (3, 9))
    m = (rng.random((3, 9)) > .3).astype(float)
    _, g = G.loss_and_grads(p, cfg, idx, tg, m)
    for k in p:
        for _ in range(3):
            ix = tuple(rng.integers(0, s) for s in p[k].shape)
            o, e = p[k][ix], 1e-5
            p[k][ix] = o + e; lp = G.loss_and_grads(p, cfg, idx, tg, m)[0]
            p[k][ix] = o - e; lm = G.loss_and_grads(p, cfg, idx, tg, m)[0]
            p[k][ix] = o
            num = (lp - lm) / (2 * e)
            assert abs(num - g[k][ix]) <= 1e-4 * (abs(num) + abs(g[k][ix])) + 1e-9, (k, ix)


def test_causal_attention_does_not_peek_at_the_future():
    cfg = G.Config(block=16, d=16, layers=2, heads=2)
    p = G.init_params(cfg, 0)
    a = np.array([[5, 6, 7, 8, 9, 10]]); b = a.copy(); b[0, 4:] = [30, 31]
    la, lb = G.forward(p, cfg, a, keep=False)[0], G.forward(p, cfg, b, keep=False)[0]
    assert np.allclose(la[0, :4], lb[0, :4], atol=1e-5)


def test_training_reduces_loss_and_save_load_roundtrips(tmp_path):
    cfg = G.Config(block=288, d=32, layers=2, heads=2)
    p = G.init_params(cfg, 0)
    opt = G.Adam(p, lr=5e-3)
    rows = load_rows()
    idx, tg, m = batch(rows, rows[:6], random.Random(0), 8, cfg.block)
    first = G.loss_and_grads(p, cfg, idx, tg, m)[0]
    for _ in range(40):
        loss, g = G.loss_and_grads(p, cfg, idx, tg, m)
        opt.step(p, g)
    assert loss < first * 0.7
    G.save(tmp_path / "m.npz", p, cfg)
    p2, cfg2 = G.load(tmp_path / "m.npz")
    assert G.generate(p, cfg, "F: a\nQ: b\nA: ", 12) == G.generate(p2, cfg2, "F: a\nQ: b\nA: ", 12)


def test_corpus_answers_only_use_numbers_from_the_facts():
    rows, rng = load_rows(), random.Random(1)
    for _ in range(400):
        prompt, ans = make_example(rows, rng)
        facts = {x.rstrip(".,") for x in re.findall(r"\d[\d,]*\.?\d*", prompt.split("\nQ:")[0])}
        q = prompt.split("\nQ: ")[1]
        for x in re.findall(r"\d[\d,]*\.?\d*", ans):
            assert x.rstrip(".,") in facts or x.rstrip(".,") in {y.rstrip(".,") for y in re.findall(r"\d+", q)}, (prompt, ans)
        assert prompt.startswith("F: ") and prompt.endswith("\nA: ") and "\n" not in ans


def test_tokenizer_roundtrip():
    s = "Motor City|1 Bedroom|p 1,100,000\nQ: ok?"
    assert G.decode(G.encode(s)) == s
