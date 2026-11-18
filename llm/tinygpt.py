"""A small GPT language model written from scratch in NumPy, with hand-written backpropagation.

No PyTorch and no other deep-learning library. Everything a GPT needs is here:
token + position embeddings, pre-LayerNorm transformer blocks (causal multi-head self-attention
and a GELU MLP), a softmax output layer, cross-entropy loss, the Adam optimiser, and sampling.
`tests/test_tinygpt.py` checks every gradient against finite differences.
"""
from __future__ import annotations
import json
import numpy as np

CHARS = "".join(chr(i) for i in range(32, 127)) + "\n"       # printable ASCII + newline
PAD = 0                                                        # id 0 is padding, ids 1.. are characters
VOCAB = len(CHARS) + 1
_ID = {c: i + 1 for i, c in enumerate(CHARS)}


def encode(text: str) -> list[int]:
    return [_ID.get(c, _ID["?"]) for c in text]


def decode(ids) -> str:
    return "".join(CHARS[i - 1] for i in ids if 0 < i <= len(CHARS))


class Config(dict):
    """vocab, block (max length), d (width), layers, heads"""
    def __init__(self, block=288, d=96, layers=3, heads=4, vocab=VOCAB):
        super().__init__(block=block, d=d, layers=layers, heads=heads, vocab=vocab)

    __getattr__ = dict.__getitem__


def init_params(cfg: Config, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    V, T, d, L = cfg.vocab, cfg.block, cfg.d, cfg.layers
    s = 0.02
    p = {"tok": rng.normal(0, s, (V, d)), "pos": rng.normal(0, s, (T, d))}
    for i in range(L):
        n = f"l{i}."
        p[n + "ln1.g"], p[n + "ln1.b"] = np.ones(d), np.zeros(d)
        p[n + "qkv.w"], p[n + "qkv.b"] = rng.normal(0, s, (d, 3 * d)), np.zeros(3 * d)
        p[n + "o.w"], p[n + "o.b"] = rng.normal(0, s / np.sqrt(2 * L), (d, d)), np.zeros(d)
        p[n + "ln2.g"], p[n + "ln2.b"] = np.ones(d), np.zeros(d)
        p[n + "fc.w"], p[n + "fc.b"] = rng.normal(0, s, (d, 4 * d)), np.zeros(4 * d)
        p[n + "pr.w"], p[n + "pr.b"] = rng.normal(0, s / np.sqrt(2 * L), (4 * d, d)), np.zeros(d)
    p["lnf.g"], p["lnf.b"] = np.ones(d), np.zeros(d)
    p["head.w"] = rng.normal(0, s, (d, V))
    return {k: v.astype(np.float32) for k, v in p.items()}


def n_params(p: dict) -> int:
    return int(sum(v.size for v in p.values()))


# ---------- building blocks (each has a forward and a backward) ----------
def ln_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-5)
    xh = (x - mu) * rstd
    return xh * g + b, (xh, rstd, g)


def ln_bwd(dy, c):
    xh, rstd, g = c
    dxh = dy * g
    dx = (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True)) * rstd
    return dx, (dy * xh).sum((0, 1)), dy.sum((0, 1))


def lin_fwd(x, w, b):
    return x @ w + b, x


def lin_bwd(dy, x, w):
    dx = dy @ w.T
    dw = x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
    return dx, dw, dy.sum((0, 1))


_C = np.sqrt(2 / np.pi)


def gelu_fwd(x):
    t = np.tanh(_C * (x + 0.044715 * x ** 3))
    return 0.5 * x * (1 + t), (x, t)


def gelu_bwd(dy, c):
    x, t = c
    return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * _C * (1 + 3 * 0.044715 * x * x))


def forward(p: dict, cfg: Config, idx: np.ndarray, keep: bool = True):
    """idx: (B,T) token ids -> logits (B,T,V). `keep` stores what backward needs."""
    B, T = idx.shape
    d, H = cfg.d, cfg.heads
    hd = d // H
    x = p["tok"][idx] + p["pos"][:T]
    cache = {"idx": idx, "layers": []}
    mask = np.tril(np.ones((T, T), dtype=bool))
    for i in range(cfg.layers):
        n = f"l{i}."
        c = {}
        h, c["ln1"] = ln_fwd(x, p[n + "ln1.g"], p[n + "ln1.b"])
        qkv, c["qkv"] = lin_fwd(h, p[n + "qkv.w"], p[n + "qkv.b"])
        q, k, v = (t.reshape(B, T, H, hd).transpose(0, 2, 1, 3) for t in np.split(qkv, 3, axis=-1))
        sc = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)
        sc = np.where(mask, sc, -1e9)
        sc = sc - sc.max(-1, keepdims=True)
        att = np.exp(sc)
        att /= att.sum(-1, keepdims=True)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        c["att"], c["q"], c["k"], c["v"] = att, q, k, v
        o, c["o"] = lin_fwd(y, p[n + "o.w"], p[n + "o.b"])
        x = x + o
        h2, c["ln2"] = ln_fwd(x, p[n + "ln2.g"], p[n + "ln2.b"])
        f, c["fc"] = lin_fwd(h2, p[n + "fc.w"], p[n + "fc.b"])
        a, c["gelu"] = gelu_fwd(f)
        m, c["pr"] = lin_fwd(a, p[n + "pr.w"], p[n + "pr.b"])
        x = x + m
        cache["layers"].append(c if keep else None)
    hf, cache["lnf"] = ln_fwd(x, p["lnf.g"], p["lnf.b"])
    cache["hf"] = hf
    return hf @ p["head.w"], cache


def loss_and_grads(p: dict, cfg: Config, idx, targets, lmask):
    """Cross-entropy over positions where lmask==1. Returns (loss, grads)."""
    B, T = idx.shape
    d, H = cfg.d, cfg.heads
    hd = d // H
    logits, cache = forward(p, cfg, idx)
    z = logits - logits.max(-1, keepdims=True)
    pr = np.exp(z)
    pr /= pr.sum(-1, keepdims=True)
    denom = max(float(lmask.sum()), 1.0)
    bi, ti = np.arange(B)[:, None], np.arange(T)[None, :]
    loss = float(-(np.log(pr[bi, ti, targets] + 1e-12) * lmask).sum() / denom)
    dl = pr
    dl[bi, ti, targets] -= 1.0
    dl *= (lmask / denom)[..., None]
    g = {}
    g["head.w"] = cache["hf"].reshape(-1, d).T @ dl.reshape(-1, dl.shape[-1])
    dx, g["lnf.g"], g["lnf.b"] = ln_bwd(dl @ p["head.w"].T, cache["lnf"])
    for i in reversed(range(cfg.layers)):
        n, c = f"l{i}.", cache["layers"][i]
        dm = dx
        da, g[n + "pr.w"], g[n + "pr.b"] = lin_bwd(dm, c["pr"], p[n + "pr.w"])
        df = gelu_bwd(da, c["gelu"])
        dh2, g[n + "fc.w"], g[n + "fc.b"] = lin_bwd(df, c["fc"], p[n + "fc.w"])
        dxl, g[n + "ln2.g"], g[n + "ln2.b"] = ln_bwd(dh2, c["ln2"])
        dx = dx + dxl
        dy, g[n + "o.w"], g[n + "o.b"] = lin_bwd(dx, c["o"], p[n + "o.w"])
        dyh = dy.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        att, q, k, v = c["att"], c["q"], c["k"], c["v"]
        datt = dyh @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ dyh
        dsc = att * (datt - (datt * att).sum(-1, keepdims=True)) / np.sqrt(hd)
        dq = dsc @ k
        dk = dsc.transpose(0, 1, 3, 2) @ q
        dqkv = np.concatenate([t.transpose(0, 2, 1, 3).reshape(B, T, d) for t in (dq, dk, dv)], axis=-1)
        dh, g[n + "qkv.w"], g[n + "qkv.b"] = lin_bwd(dqkv, c["qkv"], p[n + "qkv.w"])
        dxl, g[n + "ln1.g"], g[n + "ln1.b"] = ln_bwd(dh, c["ln1"])
        dx = dx + dxl
    g["tok"] = np.zeros_like(p["tok"])
    np.add.at(g["tok"], cache["idx"], dx)
    g["pos"] = np.zeros_like(p["pos"])
    g["pos"][:T] = dx.sum(0)
    return loss, g


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.99, wd=0.01, clip=1.0):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t, self.lr, self.b1, self.b2, self.wd, self.clip = 0, lr, b1, b2, wd, clip

    def step(self, params, grads, lr=None):
        lr = self.lr if lr is None else lr
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        sc = min(1.0, self.clip / (norm + 1e-9))
        self.t += 1
        for k, w in params.items():
            gr = grads[k] * sc
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gr
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gr * gr
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            w -= lr * (mh / (np.sqrt(vh) + 1e-8) + (self.wd * w if k.endswith(".w") else 0))
        return norm


def generate(p, cfg, prompt: str, max_new=220, temperature=0.0, top_k=8, seed=0, stop="\n"):
    """Continue `prompt` one character at a time. Stops at `stop` (default: end of line)."""
    rng = np.random.default_rng(seed)
    ids = encode(prompt)
    out = []
    for _ in range(max_new):
        ctx = np.array([ids[-cfg.block:]])
        logits = forward(p, cfg, ctx, keep=False)[0][0, -1].astype(np.float64)
        logits[PAD] = -1e9
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            top = np.argsort(logits)[-top_k:]
            pr = np.exp((logits[top] - logits[top].max()) / temperature)
            nxt = int(rng.choice(top, p=pr / pr.sum()))
        ch = CHARS[nxt - 1]
        if ch == stop:
            break
        out.append(nxt)
        ids.append(nxt)
    return decode(out)


def generate_fast(p, cfg, prompt: str, max_new=220, stop="\n"):
    """Greedy decoding with a key/value cache: same output as generate(temperature=0), far fewer operations.
    Falls back to generate() if the text would not fit in one context window."""
    ids = encode(prompt)
    if len(ids) >= cfg.block:
        return generate(p, cfg, prompt, max_new=max_new, temperature=0.0, stop=stop)
    d, H = cfg.d, cfg.heads
    hd = d // H
    logits, cache = forward(p, cfg, np.array([ids]), keep=True)
    K = [c["k"] for c in cache["layers"]]  # (1,H,T,hd)
    V = [c["v"] for c in cache["layers"]]
    last = logits[0, -1].astype(np.float64)
    out = []
    for _ in range(max_new):
        last[PAD] = -1e9
        nxt = int(last.argmax())
        ch = CHARS[nxt - 1]
        if ch == stop:
            break
        out.append(nxt)
        if len(ids) >= cfg.block - 1:  # window is full: finish with the sliding-window decoder
            rest = generate(p, cfg, prompt + decode(out), max_new=max_new - len(out), temperature=0.0, stop=stop)
            return decode(out) + rest
        t = len(ids)
        ids.append(nxt)
        x = (p["tok"][nxt] + p["pos"][t])[None, None, :]  # (1,1,d)
        for i in range(cfg.layers):
            n = f"l{i}."
            h, _ = ln_fwd(x, p[n + "ln1.g"], p[n + "ln1.b"])
            qkv, _ = lin_fwd(h, p[n + "qkv.w"], p[n + "qkv.b"])
            q, k, v = (a.reshape(1, 1, H, hd).transpose(0, 2, 1, 3) for a in np.split(qkv, 3, axis=-1))
            K[i] = np.concatenate([K[i], k], axis=2)
            V[i] = np.concatenate([V[i], v], axis=2)
            sc = (q @ K[i].transpose(0, 1, 3, 2)) / np.sqrt(hd)
            sc = sc - sc.max(-1, keepdims=True)
            att = np.exp(sc)
            att /= att.sum(-1, keepdims=True)
            y = (att @ V[i]).transpose(0, 2, 1, 3).reshape(1, 1, d)
            o, _ = lin_fwd(y, p[n + "o.w"], p[n + "o.b"])
            x = x + o
            h2, _ = ln_fwd(x, p[n + "ln2.g"], p[n + "ln2.b"])
            f, _ = lin_fwd(h2, p[n + "fc.w"], p[n + "fc.b"])
            a, _ = gelu_fwd(f)
            m, _ = lin_fwd(a, p[n + "pr.w"], p[n + "pr.b"])
            x = x + m
        hf, _ = ln_fwd(x, p["lnf.g"], p["lnf.b"])
        last = (hf @ p["head.w"])[0, 0].astype(np.float64)
    return decode(out)


def save(path, p, cfg):
    np.savez_compressed(path, __cfg=np.array(json.dumps(dict(cfg))), **p)


def load(path):
    z = np.load(path, allow_pickle=False)
    cfg = Config(**json.loads(str(z["__cfg"])))
    return {k: z[k].astype(np.float32) for k in z.files if k != "__cfg"}, cfg
