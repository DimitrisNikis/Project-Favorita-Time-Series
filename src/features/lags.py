"""Lag features: точечные лаги + агрегации."""

import pandas as pd

from src.config import LAGS


def add_point_lags(target_df, history_df, lags=None):
    """Добавляет lag_16..lag_44, lag_mean, lag_std."""
    if lags is None:
        lags = LAGS

    out = target_df.copy()
    base = history_df[["store_nbr", "item_nbr", "date", "unit_sales"]].copy()

    for L in lags:
        tmp = base.copy()
        tmp["date"] = tmp["date"] + pd.Timedelta(days=L)
        tmp = tmp.rename(columns={"unit_sales": f"lag_{L}"})
        out = out.merge(tmp, on=["store_nbr", "item_nbr", "date"], how="left")
        out[f"lag_{L}"] = out[f"lag_{L}"].fillna(0).astype("float32")

    lag_cols = [f"lag_{L}" for L in lags]
    out["lag_mean"] = out[lag_cols].mean(axis=1).astype("float32")
    out["lag_std"] = out[lag_cols].std(axis=1).fillna(0).astype("float32")
    return out
