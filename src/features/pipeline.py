"""Полный feature-engineering pipeline."""

import gc
import time

import numpy as np
import pandas as pd

from src.config import STAT_WINDOWS
from src.features.window_stats import (
    compute_extra_stats,
    compute_promo_stats,
    compute_window_stats,
)


def build_features_for_dates(target_df, history, ref_dates, items_w, stores_w, oil_w,
                              gap=16, is_test=False):
    """
    Полный feature-engineering pipeline.
    gap: отступ в днях для window stats.
      - train: gap=16 (имитируем test-time условия)
      - valid/test: gap=0 (history уже заморожена, отступ естественный)
    """
    t0 = time.time()

    print(f"  Using gap={gap} days for window stats")

    print("  [1/6] store x item sales stats...")
    si_sales = compute_window_stats(
        history, ref_dates,
        group_cols=["store_nbr", "item_nbr"],
        target_col="unit_sales",
        windows=STAT_WINDOWS,
        prefix="si_sales",
        gap=gap,
    )

    print("  [2/6] item sales stats...")
    item_sales = compute_window_stats(
        history, ref_dates,
        group_cols=["item_nbr"],
        target_col="unit_sales",
        windows=[7, 14, 30, 60],
        prefix="item_sales",
        gap=gap,
    )

    print("  [3/6] store x class sales stats...")
    sc_sales = compute_window_stats(
        history, ref_dates,
        group_cols=["store_nbr", "class"],
        target_col="unit_sales",
        windows=[7, 14, 30],
        prefix="sc_sales",
        gap=gap,
    )

    print("  [4/6] promo stats...")
    promo_st = compute_promo_stats(history, ref_dates, windows=[7, 14, 30], gap=gap)

    print("  [5/6] extra stats (days_since_last, zero_ratio)...")
    extra = compute_extra_stats(history, ref_dates, gap=gap)

    print("  [6/6] Merging...")
    out = target_df.copy()

    out = out.merge(items_w, on="item_nbr", how="left")
    out = out.merge(stores_w, on="store_nbr", how="left")
    out = out.merge(oil_w, on="date", how="left")

    out["dow"] = out["date"].dt.dayofweek.astype("int8")
    out["day"] = out["date"].dt.day.astype("int8")
    out["is_payday"] = (
        (out["day"] == 15) | (out["date"].dt.is_month_end)
    ).astype("int8")

    if len(si_sales) > 0:
        out = out.merge(si_sales, on=["date", "store_nbr", "item_nbr"], how="left")
    if len(item_sales) > 0:
        out = out.merge(item_sales, on=["date", "item_nbr"], how="left")
    if len(sc_sales) > 0:
        out = out.merge(sc_sales, on=["date", "store_nbr", "class"], how="left")
    if len(promo_st) > 0:
        out = out.merge(promo_st, on=["date", "store_nbr", "item_nbr"], how="left")
    if len(extra) > 0:
        out = out.merge(extra, on=["date", "store_nbr", "item_nbr"], how="left")

    # Diff / ratio features
    for cs, cl in [
        ("si_sales_mean_7d", "si_sales_mean_14d"),
        ("si_sales_mean_14d", "si_sales_mean_30d"),
        ("si_sales_mean_30d", "si_sales_mean_60d"),
    ]:
        if cs in out.columns and cl in out.columns:
            out[f"diff_{cs.split('_')[-1]}_{cl.split('_')[-1]}"] = out[cs] - out[cl]

    if "si_sales_mean_7d" in out.columns and "si_sales_mean_30d" in out.columns:
        out["ratio_7d_30d"] = out["si_sales_mean_7d"] / (out["si_sales_mean_30d"] + 1)
    if "item_sales_mean_7d" in out.columns and "item_sales_mean_14d" in out.columns:
        out["item_diff_7d_14d"] = out["item_sales_mean_7d"] - out["item_sales_mean_14d"]

    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].fillna(0)
    for c in out.select_dtypes(include=["float64"]).columns:
        out[c] = out[c].astype("float32")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s. Shape: {out.shape}")
    gc.collect()
    return out
