"""Read out the second-order nudge experiment.

Metric: 90-day revenue per buyer. First-order value happens before the
nudge, so it is a valid CUPED covariate and narrows the confidence interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cuped(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(n)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def readout(df: pd.DataFrame, metric: str = "ltv_90d", covariate: str = "first_order_value") -> dict:
    d = df[df["experiment_variant"].isin(["control", "treatment"])].copy()
    d[metric] = d[metric].fillna(0)
    adj = cuped(d[metric].to_numpy(), d[covariate].to_numpy())
    d["adj"] = adj
    c, t = d[d.experiment_variant == "control"], d[d.experiment_variant == "treatment"]

    raw_lift = t[metric].mean() - c[metric].mean()
    adj_lift = t["adj"].mean() - c["adj"].mean()
    p = stats.ttest_ind(t["adj"], c["adj"], equal_var=False).pvalue

    rep_c = (c["orders_30d"] >= 2).mean()
    rep_t = (t["orders_30d"] >= 2).mean()
    return {
        "buyers_control": len(c),
        "buyers_treatment": len(t),
        "mean_control": float(c[metric].mean()),
        "mean_treatment": float(t[metric].mean()),
        "lift_abs": float(adj_lift),
        "lift_pct": float(adj_lift / c[metric].mean()),
        "raw_lift_abs": float(raw_lift),
        "ci95_cuped": bootstrap_ci(c["adj"].to_numpy(), t["adj"].to_numpy()),
        "ci95_raw": bootstrap_ci(c[metric].to_numpy(), t[metric].to_numpy()),
        "p_value": float(p),
        "repeat_30d_control": float(rep_c),
        "repeat_30d_treatment": float(rep_t),
    }
