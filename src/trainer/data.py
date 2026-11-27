"""Monthly churn records over two years where the relationship slowly changes.

Early on, churn is driven mostly by support calls. A competitor launches in month 12 and from then on
data-heavy users on older plans leave more often. A model judged on a random split sees the future
during training; a model judged on the last months does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load(n: int = 24_000, months: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    month = rng.integers(0, months, n)
    df = pd.DataFrame({
        "month": month,
        "tenure_months": rng.gamma(2, 12, n).round(),
        "support_calls": rng.poisson(1.0, n),
        "data_gb": rng.lognormal(2.3, 0.6, n).round(1),
        "plan_age_years": rng.integers(0, 6, n),
        "monthly_bill": rng.normal(60, 18, n).clip(15, 160).round(2),
        "autopay": (rng.random(n) < 0.6).astype(int),
    })
    competitor = np.clip((month - 11) / 6, 0, 1)
    z = (-2.0 + 0.55 * df.support_calls - 0.03 * df.tenure_months - 0.6 * df.autopay
         + competitor * (0.06 * (df.data_gb - 10) + 0.35 * df.plan_age_years - 0.8))
    df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df
