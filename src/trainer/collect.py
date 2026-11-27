"""Gather metrics.json from every run of a Hydra multirun directory into one table."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def collect(root: Path = Path("multirun")) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(root.glob("**/metrics.json"))]
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "split"])
    return g.agg(runs=("seed", "size"), validation_auc=("validation_auc", "mean"), future_auc=("future_auc", "mean"),
                  estimate_error=("estimate_error", "mean"),
                  estimate_error_sd=("estimate_error", "std"),
                  future_auc_refit=("future_auc_refit_on_all_dev", "mean")).round(4).reset_index()


def picks(df: pd.DataFrame) -> pd.DataFrame:
    """Which model each validation scheme would pick per seed, and how it then does on the future months."""
    rows = []
    for (seed, split), g in df.groupby(["seed", "split"]):
        chosen = g.loc[g.validation_auc.idxmax()]
        best = df[(df.seed == seed)].groupby("model").future_auc_refit_on_all_dev.first().idxmax()
        rows.append({"seed": seed, "split": split, "picked": chosen.model, "best_on_future": best,
                     "future_auc_of_pick": chosen.future_auc_refit_on_all_dev})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    d = collect()
    print(summary(d).to_string(index=False))
    print(picks(d).to_string(index=False))
