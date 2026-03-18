"""Оконные статистики: multi-window mean/std/median, promo stats, extra stats."""

import pandas as pd


def compute_window_stats(history, ref_dates, group_cols, target_col, windows, prefix, gap=16):
    """
    Winner's approach: multi-window статистики (mean, std, median) для каждой ref_date.
    gap=16 — отступ чтобы train фичи имели ту же «протухлость» что и test/valid.
    """
    results = []
    for ref_date in ref_dates:
        cutoff = ref_date - pd.Timedelta(days=gap)
        hist = history[history["date"] <= cutoff]
        if len(hist) == 0:
            continue
        days_ago = (cutoff - hist["date"]).dt.days

        row_features = []
        for w in windows:
            w_data = hist.loc[days_ago <= w]
            if len(w_data) == 0:
                continue
            stats = (
                w_data.groupby(group_cols, observed=True)[target_col]
                .agg(
                    **{
                        f"{prefix}_mean_{w}d": "mean",
                        f"{prefix}_std_{w}d": "std",
                        f"{prefix}_median_{w}d": "median",
                    }
                )
                .reset_index()
            )
            row_features.append(stats)

        if not row_features:
            continue
        merged = row_features[0]
        for rf in row_features[1:]:
            merged = merged.merge(rf, on=group_cols, how="outer")
        merged["date"] = ref_date
        results.append(merged)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def compute_promo_stats(history, ref_dates, windows, prefix="si_promo", gap=16):
    """Промо-статистики по окнам для store x item."""
    results = []
    for ref_date in ref_dates:
        cutoff = ref_date - pd.Timedelta(days=gap)
        hist = history[history["date"] <= cutoff]
        if len(hist) == 0:
            continue
        days_ago = (cutoff - hist["date"]).dt.days

        row_features = []
        for w in windows:
            w_data = hist.loc[days_ago <= w]
            if len(w_data) == 0:
                continue
            stats = (
                w_data.groupby(["store_nbr", "item_nbr"], observed=True)["onpromotion"]
                .agg(**{f"{prefix}_mean_{w}d": "mean", f"{prefix}_sum_{w}d": "sum"})
                .reset_index()
            )
            row_features.append(stats)

        if not row_features:
            continue
        merged = row_features[0]
        for rf in row_features[1:]:
            merged = merged.merge(rf, on=["store_nbr", "item_nbr"], how="outer")
        merged["date"] = ref_date
        results.append(merged)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def compute_extra_stats(history, ref_dates, gap=16):
    """days_since_last + zero_ratio для store x item."""
    results = []
    for ref_date in ref_dates:
        cutoff = ref_date - pd.Timedelta(days=gap)
        hist = history[history["date"] <= cutoff]
        if len(hist) == 0:
            continue
        days_ago = (cutoff - hist["date"]).dt.days

        last = hist.groupby(["store_nbr", "item_nbr"], observed=True)["date"].max().reset_index()
        last["si_days_since_last"] = (cutoff - last["date"]).dt.days
        feat = last[["store_nbr", "item_nbr", "si_days_since_last"]]

        for w in [7, 14, 30]:
            w_data = hist.loc[days_ago <= w]
            if len(w_data) == 0:
                continue
            zr = (
                w_data.groupby(["store_nbr", "item_nbr"], observed=True)["unit_sales"]
                .agg(**{f"si_zero_ratio_{w}d": lambda x: (x == 0).mean()})
                .reset_index()
            )
            feat = feat.merge(zr, on=["store_nbr", "item_nbr"], how="outer")

        feat["date"] = ref_date
        results.append(feat)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)
