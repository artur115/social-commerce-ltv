import numpy as np
import pandas as pd

from ltv import experiment, model


def test_top_decile_capture_perfect_ranking():
    y = np.arange(100, dtype=float)
    assert model.top_decile_capture(y, y) == y[-10:].sum() / y.sum()


def test_cuped_keeps_mean_and_cuts_variance():
    rng = np.random.default_rng(1)
    x = rng.normal(50, 10, 5000)
    y = 2 * x + rng.normal(0, 5, 5000)
    adj = experiment.cuped(y, x)
    assert abs(adj.mean() - y.mean()) < 1e-9
    assert adj.var() < y.var() / 10


def test_readout_detects_planted_lift():
    rng = np.random.default_rng(2)
    n = 4000
    variant = np.where(rng.random(n) < 0.5, "treatment", "control")
    first = rng.lognormal(3.5, 0.4, n)
    ltv = first + rng.gamma(2, 15, n) + np.where(variant == "treatment", 8, 0)
    df = pd.DataFrame({"experiment_variant": variant, "first_order_value": first, "ltv_90d": ltv,
                       "orders_30d": rng.integers(1, 3, n)})
    r = experiment.readout(df)
    assert r["ci95_cuped"][0] > 0 and 4 < r["lift_abs"] < 12
