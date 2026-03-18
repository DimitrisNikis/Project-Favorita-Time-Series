"""Dense grid: восстановление нулевых продаж (implicit zeros)."""

import numpy as np
import pandas as pd


def make_dense_grid(start_date, end_date, stores, items):
    """Создаёт декартово произведение dates × stores × items."""
    dates = pd.date_range(start_date, end_date, freq="D")
    g = pd.MultiIndex.from_product(
        [dates, stores, items], names=["date", "store_nbr", "item_nbr"]
    ).to_frame(index=False)
    g["store_nbr"] = g["store_nbr"].astype("int16")
    g["item_nbr"] = g["item_nbr"].astype("int32")
    return g


def attach_target_and_promo(grid, obs_df):
    """Мержит наблюдения на grid, заполняет пропуски нулями."""
    out = grid.merge(
        obs_df[["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]],
        on=["date", "store_nbr", "item_nbr"],
        how="left",
    )
    out["unit_sales"] = out["unit_sales"].fillna(0).clip(lower=0).astype("float32")
    out["onpromotion"] = out["onpromotion"].fillna(0).astype("int8")
    return out
