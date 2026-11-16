"""Training data for the mini GPT, made from the app's own ranking table.

Design: the app looks up the matching rows (its normal search) and writes them as a facts line.
The model's job is to turn "facts + question" into a clear sentence and copy the numbers exactly.
Small models are good at copying, and bad at remembering 229 rows, so this is the honest task for them.
"""
from __future__ import annotations
import csv
import hashlib
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ascii(s: str) -> str:
    return s.encode("ascii", "ignore").decode().strip()


def load_rows(path=ROOT / "data" / "ranking.csv"):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            g = float(r["growth_12m"]) if r["growth_12m"] not in ("", "nan", "NaN") else None
            rows.append(dict(area=_ascii(r["area"]), type=_ascii(r["property_type"]), price=int(round(float(r["price"]))),
                             rent=int(round(float(r["rent"]))), net=float(r["net_yield"]) * 100,
                             gross=float(r["gross_yield"]) * 100, growth=None if g is None else g * 100, rank=i + 1))
    for r in rows:
        r["of"] = len(rows)
    return rows


def is_holdout(area: str, frac: float) -> bool:
    return frac > 0 and int(hashlib.md5(area.encode()).hexdigest(), 16) % 1000 < frac * 1000


def facts_line(r) -> str:
    gr = "na" if r["growth"] is None else f"{r['growth']:.0f}"
    return (f"F: {r['area']}|{r['type']}|p {r['price']:,}|r {r['rent']:,}|n {r['net']:.1f}|g {r['gross']:.1f}|"
            f"y {gr}|#{r['rank']}/{r['of']}")


def best_line(label, top) -> str:
    return f"F: best {label}|" + "|".join(f"{t['area']} {t['net']:.1f}" for t in top)


QS = {
    "summary": ["tell me about {a} {t}", "{a} {t} details", "summary of {t} in {a}"],
    "price": ["how much is a {t} in {a}", "price of {a} {t}", "what does a {t} in {a} cost"],
    "rent": ["what rent can i get for a {t} in {a}", "rent for {a} {t}", "how much rent for {t} in {a}"],
    "yield": ["what is the net yield of {t} in {a}", "{a} {t} yield", "rental yield for {a} {t}"],
    "growth": ["will prices grow in {a} for a {t}", "growth estimate for {a} {t}", "price growth {a} {t}"],
    "good": ["is {a} {t} a good buy", "should i buy a {t} in {a}", "is a {t} in {a} worth it"],
}
BEST_QS = ["best yield for {t}", "which area has the best net yield for a {t}", "top areas for {t}"]
ANY_QS = ["best yield", "which area has the best net yield", "top areas for rental yield"]


def answer(intent, r) -> str:
    a, t, gr = r["area"], r["type"], r["growth"]
    g = "n/a" if gr is None else f"{gr:.0f}%"
    if intent == "summary":
        return f"{t} in {a}: AED {r['price']:,}, rent AED {r['rent']:,}/yr, net {r['net']:.1f}%, gross {r['gross']:.1f}%, growth {g}."
    if intent == "price":
        return f"A {t} in {a} costs about AED {r['price']:,}."
    if intent == "rent":
        return f"A {t} in {a} rents for about AED {r['rent']:,} a year."
    if intent == "yield":
        return f"Net yield is {r['net']:.1f}% after costs. Gross yield is {r['gross']:.1f}%."
    if intent == "growth":
        if gr is None:
            return f"Not enough price history to estimate growth for {a} {t}."
        return f"Estimated price growth over the next year is {gr:.0f}%. It is a trend, not a promise."
    return f"{a} {t} ranks #{r['rank']} of {r['of']}: net yield {r['net']:.1f}%, growth {g}. Check the building first."


def best_answer(label, top) -> str:
    t = [f"{x['area']} at {x['net']:.1f}%" for x in top]
    what = f"for {label}" if label != "any" else ""
    s = f"The best net yield {what}".replace("  ", " ").strip() + f" is {t[0]}"
    if len(t) > 1:
        s += f", then {t[1]}" + (f" and {t[2]}" if len(t) > 2 else "")
    return s + "."


TYPES = ["Studio", "1 Bedroom", "2 Bedroom", "3 Bedroom", "4 Bedroom", "5 Bedroom"]
_WORDS = None


def synth_row(rows, rng: random.Random):
    """A made-up row with a random name and random numbers. Training on these makes memorising impossible,
    so the model has to learn the real skill: copying the facts exactly."""
    global _WORDS
    if _WORDS is None:
        _WORDS = sorted({w for r in rows for w in r["area"].replace("-", " ").split()} | {"Gardens", "Heights", "Bay", "Park", "Oasis", "Springs", "Walk"})
    n = rng.choice([1, 2, 2, 3, 4])
    name = " ".join(rng.choice(_WORDS) for _ in range(n))
    if rng.random() < 0.15:
        name += rng.choice([" 1", " 2", " 3", " Phase 2", " District 4"])
    if rng.random() < 0.1:
        name = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(4, 9))).title()
    step = rng.choice([1, 1000, 1000, 5000, 10000])
    price = int(round(10 ** rng.uniform(5.4, 7.1) / step) * step)
    gross = round(rng.uniform(2.2, 9.5), 1)
    m = rng.choice([1, 500, 1000])
    rent = max(int(round(price * gross / 100 / m)) * m, 1000)
    return dict(area=name, type=rng.choice(TYPES), price=price, rent=rent, net=round(gross * rng.uniform(.55, .85), 1),
                gross=gross, growth=None if rng.random() < .15 else float(rng.randint(-8, 22)),
                rank=rng.randint(1, 229), of=229)


def make_example(rows, rng: random.Random, pool=None, synth: float = 0.7):
    """Returns (prompt, answer). `pool` = real rows allowed for training (default all). `synth` = share of made-up rows."""
    pool = pool or rows
    if rng.random() < 0.2:                                   # "best" questions use several rows
        typ = rng.choice(sorted({r["type"] for r in pool}) + ["any"])
        sel = [r for r in pool if typ == "any" or r["type"] == typ]
        if rng.random() < synth:                             # made-up leaderboard
            sel = [synth_row(rows, rng) for _ in range(3)]
            for x in sel:
                x["type"] = typ if typ != "any" else x["type"]
        top = sorted(sel, key=lambda r: -r["net"])[:3]
        q = rng.choice(ANY_QS if typ == "any" else BEST_QS).format(t=typ.lower())
        return f"{best_line(typ, top)}\nQ: {q}\nA: ", best_answer(typ, top)
    r = synth_row(rows, rng) if rng.random() < synth else rng.choice(pool)
    intent = rng.choice(list(QS))
    q = rng.choice(QS[intent]).format(a=r["area"].lower(), t=r["type"].lower())
    return f"{facts_line(r)}\nQ: {q}\nA: ", answer(intent, r)
