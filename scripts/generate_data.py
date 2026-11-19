"""Generate a synthetic GA4-style event export for a social-commerce marketplace.

The data is fake but shaped like the real thing: users arrive from social
channels, come back through retargeting, email and live shopping, and buy
with different habits. Several effects are planted so the analysis has
something true to find:

* Live-shopping buyers place smaller first orders but repeat far more often.
* Deal seekers from short-video ads buy once and rarely return.
* A second order inside 30 days is the strongest sign of a high-value buyer.
* From 2025-06-01, new social buyers are split 50/50 into an experiment.
  The treatment gets a second-order nudge (coupon plus live-stream invite),
  which raises the chance of a second order for some buyer types.
* Raw exports contain duplicate event deliveries and mixed-case sources.

Usage: python scripts/generate_data.py [--users 30000] [--seed 42] [--out data/raw]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

START = datetime(2025, 1, 1, tzinfo=timezone.utc)
ACQ_END = datetime(2025, 9, 30, tzinfo=timezone.utc)
OBS_END = datetime(2026, 3, 31, tzinfo=timezone.utc)
EXPERIMENT_START = datetime(2025, 6, 1, tzinfo=timezone.utc)

# Acquisition channel -> (source, medium, share of new users)
CHANNELS = {
    "tiktok": ("tiktok", "paid_social", 0.18),
    "instagram": ("instagram", "paid_social", 0.15),
    "youtube": ("youtube", "paid_social", 0.09),
    "facebook": ("facebook", "paid_social", 0.08),
    "influencer": ("creator_network", "influencer", 0.11),
    "live_stream": ("live", "live_shopping", 0.08),
    "paid_search": ("google", "cpc", 0.15),
    "organic": ("google", "organic", 0.10),
    "referral": ("friend_invite", "referral", 0.06),
}
SOCIAL = {"tiktok", "instagram", "youtube", "facebook", "influencer", "live_stream"}

# Buyer types: (repeat propensity, first-order AOV mean, later AOV mean, sessions per month)
TYPES = {
    "deal_seeker": (0.10, 24.0, 21.0, 1.6),
    "browser": (0.05, 38.0, 35.0, 2.2),
    "loyal": (0.34, 46.0, 52.0, 2.6),
    "live_fan": (0.42, 29.0, 36.0, 3.1),
}
# Probability of each buyer type, by acquisition channel
TYPE_MIX = {
    "tiktok": [0.55, 0.25, 0.10, 0.10],
    "instagram": [0.35, 0.30, 0.20, 0.15],
    "youtube": [0.25, 0.35, 0.25, 0.15],
    "facebook": [0.35, 0.30, 0.25, 0.10],
    "influencer": [0.25, 0.25, 0.25, 0.25],
    "live_stream": [0.15, 0.10, 0.20, 0.55],
    "paid_search": [0.30, 0.30, 0.35, 0.05],
    "organic": [0.20, 0.35, 0.40, 0.05],
    "referral": [0.20, 0.25, 0.45, 0.10],
}
# Return-visit sources: (source, medium, weight)
RETURN_SOURCES = [
    ("facebook", "paid_social", 0.14),
    ("instagram", "paid_social", 0.14),
    ("tiktok", "paid_social", 0.08),
    ("newsletter", "email", 0.18),
    ("(direct)", "(none)", 0.20),
    ("google", "organic", 0.10),
    ("google", "cpc", 0.08),
    ("live", "live_shopping", 0.08),
]
# Treatment lift on repeat propensity, by buyer type
NUDGE_LIFT = {"deal_seeker": 0.05, "browser": 0.01, "loyal": 0.06, "live_fan": 0.14}


def _ts(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def generate(n_users: int = 30_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(CHANNELS)
    shares = np.array([CHANNELS[c][2] for c in names])
    shares = shares / shares.sum()
    type_names = list(TYPES)
    ret_src = [(s, m) for s, m, _ in RETURN_SOURCES]
    ret_w = np.array([w for *_, w in RETURN_SOURCES])
    ret_w = ret_w / ret_w.sum()
    span = (ACQ_END - START).total_seconds()

    rows: list[tuple] = []
    order_no = 0

    for u in range(n_users):
        uid = f"{rng.integers(10**9, 10**10)}.{rng.integers(10**9, 10**10)}"
        channel = names[rng.choice(len(names), p=shares)]
        btype = type_names[rng.choice(4, p=TYPE_MIX[channel])]
        repeat_p, aov1, aov_n, spm = TYPES[btype]
        first_seen = START + timedelta(seconds=float(rng.uniform(0, span)))

        variant = None
        if channel in SOCIAL and first_seen >= EXPERIMENT_START:
            variant = "treatment" if rng.random() < 0.5 else "control"

        # Session times: first visit, then a Poisson process with decaying interest
        t = first_seen
        session_times = [t]
        churn_rate = 1 / (60 if btype in ("loyal", "live_fan") else 25)
        while True:
            gap_days = rng.exponential(30 / spm)
            t = t + timedelta(days=float(gap_days))
            if t > OBS_END or rng.random() < churn_rate * gap_days:
                break
            session_times.append(t)

        orders = 0
        last_order_at = None
        for i, st in enumerate(session_times):
            if i == 0:
                src, med = CHANNELS[channel][:2]
                campaign = f"{channel}_{first_seen:%Y%m}_prospecting"
            else:
                src, med = ret_src[rng.choice(len(ret_src), p=ret_w)]
                campaign = "(not set)" if med in ("(none)", "organic") else f"{src}_retarget"
                # Live fans come back through live shows more often
                if btype == "live_fan" and rng.random() < 0.35:
                    src, med, campaign = "live", "live_shopping", "weekly_live_show"

            sid = int(st.timestamp())
            ev_t = st
            raw_src = src.upper() if rng.random() < 0.03 else src  # casing drift
            base = (uid, sid, raw_src, med, campaign, variant)
            rows.append((_ts(ev_t), "session_start", *base, None, None))
            n_views = 1 + rng.poisson(2.5)
            for _ in range(n_views):
                ev_t += timedelta(seconds=float(rng.uniform(10, 120)))
                rows.append((_ts(ev_t), "view_item", *base, None, None))

            # Purchase probability for this session
            if orders == 0:
                p_buy = 0.32 if i == 0 else 0.12
            else:
                p_buy = repeat_p
                if variant == "treatment" and orders == 1 and last_order_at and (st - last_order_at).days <= 30:
                    p_buy += NUDGE_LIFT[btype] * 2.2
                if med == "live_shopping":
                    p_buy *= 1.25
            if rng.random() < min(p_buy, 0.95):
                ev_t += timedelta(seconds=float(rng.uniform(20, 90)))
                rows.append((_ts(ev_t), "add_to_cart", *base, None, None))
                ev_t += timedelta(seconds=float(rng.uniform(20, 200)))
                rows.append((_ts(ev_t), "begin_checkout", *base, None, None))
                mean = aov1 if orders == 0 else aov_n
                value = float(np.round(rng.lognormal(np.log(mean), 0.45), 2))
                if variant == "treatment" and orders == 1:
                    value = round(value * 0.9, 2)  # coupon on the nudged order
                order_no += 1
                ev_t += timedelta(seconds=float(rng.uniform(10, 60)))
                rows.append((_ts(ev_t), "purchase", *base, value, f"T{order_no:07d}"))
                orders += 1
                last_order_at = st

    cols = ["event_timestamp", "event_name", "user_pseudo_id", "ga_session_id", "source", "medium",
            "campaign", "experiment_variant", "purchase_revenue_usd", "transaction_id"]
    df = pd.DataFrame(rows, columns=cols)

    # Duplicate deliveries (about 1.5% of events), as raw exports often have
    dups = df.sample(frac=0.015, random_state=seed)
    df = pd.concat([df, dups], ignore_index=True).sort_values("event_timestamp", kind="stable")
    df["event_date"] = pd.to_datetime(df["event_timestamp"], unit="us", utc=True).dt.strftime("%Y%m%d")
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/raw")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = generate(a.users, a.seed)
    df.to_parquet(out / "events.parquet", index=False)
    print(f"wrote {len(df):,} events for {df.user_pseudo_id.nunique():,} users to {out / 'events.parquet'}")


if __name__ == "__main__":
    main()
