"""Multi-touch attribution: rule-based models and a first-order Markov model.

Every function takes a touchpath table with one row per touch:
    transaction_id, revenue, ordered_at, channel, session_start_at, touch_number
and returns revenue credited to each channel. Credits always sum to the
total revenue of the orders, so models can be compared like for like.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _weights(paths: pd.DataFrame, model: str, half_life_days: float = 7.0) -> pd.Series:
    g = paths.groupby("transaction_id")
    n = g["touch_number"].transform("max")
    k = paths["touch_number"]
    if model == "last_click":
        return (k == n).astype(float)
    if model == "first_click":
        return (k == 1).astype(float)
    if model == "linear":
        return 1.0 / n
    if model == "time_decay":
        age = (paths["ordered_at"] - paths["session_start_at"]).dt.total_seconds() / 86400
        w = np.power(0.5, age / half_life_days)
        return w / w.groupby(paths["transaction_id"]).transform("sum")
    if model == "position_based":
        # 40% first, 40% last, 20% split across the middle. One touch gets 100%,
        # two touches get 50/50.
        w = pd.Series(0.0, index=paths.index)
        w[n == 1] = 1.0
        w[(n == 2)] = 0.5
        many = n >= 3
        w[many & (k == 1)] = 0.4
        w[many & (k == n)] = 0.4
        mid = many & (k > 1) & (k < n)
        w[mid] = 0.2 / (n[mid] - 2)
        return w
    raise ValueError(f"unknown model: {model}")


def rule_based(paths: pd.DataFrame, model: str) -> pd.Series:
    """Revenue credited per channel under a rule-based model."""
    credit = paths["revenue"] * _weights(paths, model)
    return credit.groupby(paths["channel"]).sum().rename(model)


def _transition_probs(sequences: list[list[str]]) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1
    return {a: {b: c / sum(nxt.values()) for b, c in nxt.items()} for a, nxt in counts.items()}


def _conversion_prob(probs: dict[str, dict[str, float]], removed: str | None = None, iters: int = 200) -> float:
    """P(reach 'conv' from 'start'), solved by value iteration. A removed channel sends traffic to 'null'."""
    states = set(probs) | {b for nxt in probs.values() for b in nxt}
    p = {s: 0.0 for s in states}
    p["conv"] = 1.0
    for _ in range(iters):
        new = dict(p)
        for s, nxt in probs.items():
            if s == removed:
                new[s] = 0.0
                continue
            new[s] = sum(q * (0.0 if b == removed else p[b]) for b, q in nxt.items())
        new["conv"], new["null"] = 1.0, 0.0
        p = new
    return p.get("start", 0.0)


def markov(paths: pd.DataFrame, non_converting: list[list[str]] | None = None) -> pd.Series:
    """First-order Markov attribution by removal effect.

    `non_converting` are channel sequences that did not end in an order.
    Without them every path converts and removal effects measure how often a
    channel appears on the way to an order.
    """
    seqs = [["start", *g.sort_values("touch_number")["channel"], "conv"] for _, g in paths.groupby("transaction_id")]
    for s in non_converting or []:
        seqs.append(["start", *s, "null"])
    probs = _transition_probs(seqs)
    base = _conversion_prob(probs)
    channels = sorted(paths["channel"].unique())
    effects = pd.Series({c: max(0.0, 1 - _conversion_prob(probs, removed=c) / base) for c in channels})
    share = effects / effects.sum()
    return (share * paths.drop_duplicates("transaction_id")["revenue"].sum()).rename("markov")


def compare(paths: pd.DataFrame, non_converting: list[list[str]] | None = None) -> pd.DataFrame:
    """All models side by side, as revenue per channel."""
    models = ["last_click", "first_click", "linear", "time_decay", "position_based"]
    out = pd.concat([rule_based(paths, m) for m in models] + [markov(paths, non_converting)], axis=1)
    return out.fillna(0.0).sort_values("markov", ascending=False)
