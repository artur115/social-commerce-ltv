"""Predict 180-day buyer value from the first 30 days, and segment buyers early."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

NUMERIC = ["first_order_value", "orders_30d", "ltv_30d", "sessions_30d", "channels_30d",
           "live_sessions_30d", "email_sessions_30d"]


def features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC].fillna(0).astype(float)
    X["second_order_in_30d"] = (df["days_to_second_order"].fillna(999) <= 30).astype(float)
    return X.join(pd.get_dummies(df["acquisition_channel"], prefix="acq", dtype=float))


def time_split(df: pd.DataFrame, test_months: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier cohorts and test on the latest ones, as you would in production."""
    months = sorted(df["cohort_month"].unique())
    cut = months[-test_months]
    return df[df["cohort_month"] < cut], df[df["cohort_month"] >= cut]


def top_decile_capture(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of all 180-day revenue held by the 10% of buyers the model ranks highest."""
    k = max(1, len(y_pred) // 10)
    idx = np.argsort(-y_pred)[:k]
    return float(y_true[idx].sum() / y_true.sum())


def fit_and_evaluate(df: pd.DataFrame, seed: int = 0) -> dict:
    train, test = time_split(df)
    Xtr, Xte = features(train), features(test).reindex(columns=features(train).columns, fill_value=0.0)
    ytr, yte = train["ltv_180d"].to_numpy(), test["ltv_180d"].to_numpy()

    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=seed)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    # Baseline: average 180-day value of the user's acquisition channel in training data
    by_channel = train.groupby("acquisition_channel")["ltv_180d"].mean()
    naive = test["acquisition_channel"].map(by_channel).fillna(ytr.mean()).to_numpy()

    return {
        "train_buyers": len(train),
        "test_buyers": len(test),
        "mae_model": float(np.mean(np.abs(pred - yte))),
        "mae_channel_avg": float(np.mean(np.abs(naive - yte))),
        "top10_capture_model": top_decile_capture(yte, pred),
        "top10_capture_channel_avg": top_decile_capture(yte, naive + np.random.default_rng(seed).normal(0, 1e-6, len(naive))),
        "model": model,
        "columns": list(Xtr.columns),
    }


def segment(df: pd.DataFrame, k: int = 4, seed: int = 0) -> pd.DataFrame:
    """Cluster buyers on early behaviour and name each cluster by what stands out."""
    cols = ["first_order_value", "orders_30d", "sessions_30d", "live_sessions_30d", "email_sessions_30d"]
    X = StandardScaler().fit_transform(df[cols].fillna(0))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    out = df.assign(segment=labels)
    prof = out.groupby("segment")[cols].mean()
    # Name each cluster by its most distinctive trait, each name used once
    names: dict[int, str] = {}
    for col, label in [("live_sessions_30d", "Live-show regulars"), ("orders_30d", "Fast repeaters"),
                       ("first_order_value", "Big first basket")]:
        rest = prof.drop(index=list(names))
        names[int(rest[col].idxmax())] = label
    for s in prof.index:
        names.setdefault(int(s), "One-and-done")
    out["segment"] = out["segment"].map(names)
    return out
