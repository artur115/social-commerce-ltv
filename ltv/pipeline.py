"""Run everything: SQL models, attribution, LTV model, segments, experiment, charts and report.

Usage: python -m ltv.pipeline [--raw data/raw/events.parquet] [--out report]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ltv import attribution, experiment, model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, GRID = "#111015", "#777286", "#e6e2f2"
PALETTE = ["#5b4fe6", "#d0688f", "#3f9fb5", "#a9a0d2", "#e0a458", "#6aa874", "#8c8799", "#c9b8ff", "#2f2a6b", "#f1b5c9"]


def run_sql(con: duckdb.DuckDBPyConnection, raw: Path) -> None:
    con.execute(f"create or replace view raw_events as select * from read_parquet('{raw.as_posix()}')")
    for f in sorted((ROOT / "sql").glob("*.sql")):
        con.execute(f.read_text())


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_ltv_curves(curves: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    order = curves[curves.day == 180].sort_values("avg_cum_revenue", ascending=False)["acquisition_channel"]
    for i, ch in enumerate(order):
        d = curves[curves.acquisition_channel == ch]
        ax.plot(d.day, d.avg_cum_revenue, color=PALETTE[i % len(PALETTE)], linewidth=2.2 if i < 3 else 1.3, label=ch)
    _style(ax, "Revenue per buyer by days since first order")
    ax.set_xlabel("Days since first order", color=MUTED)
    ax.set_ylabel("Cumulative revenue (USD)", color=MUTED)
    ax.legend(frameon=False, fontsize=8, ncol=2, title="Acquisition channel", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_attribution(comp: pd.DataFrame, path: Path) -> None:
    share = comp / comp.sum()
    cols = ["last_click", "linear", "markov"]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    x = range(len(share))
    w = 0.27
    for i, c in enumerate(cols):
        ax.bar([p + (i - 1) * w for p in x], share[c] * 100, width=w, color=PALETTE[i], label=c.replace("_", " "))
    ax.set_xticks(list(x), share.index, rotation=30, ha="right")
    _style(ax, "Share of revenue credited to each channel")
    ax.set_ylabel("% of revenue", color=MUTED)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_segments(seg: pd.DataFrame, path: Path) -> None:
    s = seg.groupby("segment").agg(buyers=("user_id", "count"), ltv=("ltv_180d", "mean")).sort_values("ltv")
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.barh(s.index, s.ltv, color=PALETTE[0])
    for i, (v, n) in enumerate(zip(s.ltv, s.buyers)):
        ax.text(v + 1, i, f"${v:,.0f}  ·  {n:,} buyers", va="center", fontsize=9, color=MUTED)
    _style(ax, "180-day revenue per buyer, by early-behaviour segment")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, s.ltv.max() * 1.45)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_experiment(r: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    lo, hi = r["ci95_raw"]
    clo, chi = r["ci95_cuped"]
    ax.errorbar([r["raw_lift_abs"]], [1], xerr=[[r["raw_lift_abs"] - lo], [hi - r["raw_lift_abs"]]], fmt="o", color=PALETTE[3], capsize=6, linewidth=2)
    ax.errorbar([r["lift_abs"]], [0], xerr=[[r["lift_abs"] - clo], [chi - r["lift_abs"]]], fmt="o", color=PALETTE[0], capsize=6, linewidth=2)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_yticks([0, 1], ["CUPED-adjusted", "Raw difference"])
    ax.set_ylim(-0.7, 1.7)
    _style(ax, "Second-order nudge: lift in 90-day revenue per buyer (95% CI)")
    ax.set_xlabel("USD per buyer, treatment minus control", color=MUTED)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/events.parquet"))
    ap.add_argument("--out", default=str(ROOT / "report"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    run_sql(con, Path(a.raw))

    paths = con.sql("select * from touchpaths").df()
    paths["ordered_at"] = pd.to_datetime(paths["ordered_at"])
    paths["session_start_at"] = pd.to_datetime(paths["session_start_at"])
    non_conv = con.sql("""
        select list(channel order by session_start_at)[1:10] as seq
        from sessions where user_id not in (select user_id from orders) group by user_id
    """).df()["seq"].tolist()
    comp = attribution.compare(paths, [list(s) for s in non_conv])

    ltv = con.sql("select * from user_ltv").df()
    curves = con.sql("select * from ltv_curves").df()
    fit = model.fit_and_evaluate(ltv)
    seg = model.segment(ltv)
    exp = experiment.readout(ltv)

    second = ltv.assign(fast=ltv["days_to_second_order"].fillna(999) <= 30).groupby("fast")["ltv_180d"].mean()
    by_channel = (ltv.groupby("acquisition_channel")
                     .agg(buyers=("user_id", "count"), first_order=("first_order_value", "mean"),
                          ltv_180d=("ltv_180d", "mean"), repeat_rate=("orders_180d", lambda s: (s >= 2).mean()))
                     .sort_values("ltv_180d", ascending=False))

    chart_ltv_curves(curves, out / "ltv_curves.png")
    chart_attribution(comp, out / "attribution.png")
    chart_segments(seg, out / "segments.png")
    chart_experiment(exp, out / "experiment.png")

    results = {
        "events": int(con.sql("select count(*) from raw_events").fetchall()[0][0]),
        "events_after_dedup": int(con.sql("select count(*) from stg_events").fetchall()[0][0]),
        "buyers_with_180d": len(ltv),
        "ltv_180d_fast_second_order": float(second.get(True, 0)),
        "ltv_180d_other": float(second.get(False, 0)),
        "model": {k: v for k, v in fit.items() if k not in ("model", "columns")},
        "experiment": exp,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "by_channel.csv").write_text(by_channel.round(3).to_csv())
    (out / "attribution.csv").write_text(comp.round(2).to_csv())
    print(json.dumps(results, indent=2, default=str))
    print(by_channel.round(2).to_string())
    print((comp / comp.sum() * 100).round(1).to_string())
    print(seg.groupby("segment").agg(buyers=("user_id", "count"), ltv=("ltv_180d", "mean")).round(1).to_string())


if __name__ == "__main__":
    main()
