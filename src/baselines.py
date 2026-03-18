"""Статистические бейзлайны: Naive, SeasonalNaive(7), AutoETS(7), AutoTheta(7)."""

import gc

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta, Naive, SeasonalNaive

from src.config import H, MAX_SERIES
from src.metrics import bias_me, nwrmsle, wmape


def run_baselines(train_dense_fe, valid_dense_fe, h=H, max_series=MAX_SERIES):
    """
    Запускает бейзлайны и возвращает DataFrame с метриками.

    Returns:
        pd.DataFrame с колонками [model, NWRMSLE, WMAPE, BIAS]
    """
    tr = train_dense_fe.copy()
    va = valid_dense_fe.copy()

    # ── Naive: последнее наблюдение ──────────────────────────
    last_obs = (
        tr.sort_values("date")
        .groupby(["store_nbr", "item_nbr"], as_index=False)["unit_sales"]
        .last()
        .rename(columns={"unit_sales": "pred_naive"})
    )
    item_mean = (
        tr.groupby("item_nbr", as_index=False)["unit_sales"]
        .mean()
        .rename(columns={"unit_sales": "item_mean"})
    )
    global_mean = float(tr["unit_sales"].mean())

    pred = va[["date", "store_nbr", "item_nbr", "unit_sales", "perishable"]].copy()
    pred = pred.merge(last_obs, on=["store_nbr", "item_nbr"], how="left")
    pred = pred.merge(item_mean, on="item_nbr", how="left")
    pred["pred_naive"] = (
        pred["pred_naive"].fillna(pred["item_mean"]).fillna(global_mean).clip(lower=0)
    )

    # ── StatsForecast: SeasonalNaive, AutoETS, AutoTheta ─────
    tr_sf = tr[["date", "store_nbr", "item_nbr", "unit_sales"]].copy()
    tr_sf["unique_id"] = (
        tr_sf["store_nbr"].astype(str) + "_" + tr_sf["item_nbr"].astype(str)
    )

    active_ids = (
        tr_sf.groupby("unique_id", as_index=False)["unit_sales"]
        .sum()
        .sort_values("unit_sales", ascending=False)
        .head(max_series)["unique_id"]
    )
    sf_train = (
        tr_sf[tr_sf["unique_id"].isin(active_ids)]
        .rename(columns={"date": "ds", "unit_sales": "y"})[["unique_id", "ds", "y"]]
    )

    sf = StatsForecast(
        models=[
            SeasonalNaive(season_length=7),
            AutoETS(season_length=7),
            AutoTheta(season_length=7),
        ],
        freq="D",
        n_jobs=-1,
        fallback_model=Naive(),
    )
    fcst = sf.forecast(df=sf_train, h=h).reset_index()

    snaive_col = [c for c in fcst.columns if "seasonalnaive" in c.lower()][0]
    ets_col = [c for c in fcst.columns if "autoets" in c.lower()][0]
    theta_col = [c for c in fcst.columns if "autotheta" in c.lower()][0]

    va_ids = pred[["date", "store_nbr", "item_nbr"]].copy()
    va_ids["unique_id"] = (
        va_ids["store_nbr"].astype(str) + "_" + va_ids["item_nbr"].astype(str)
    )
    va_ids = va_ids.rename(columns={"date": "ds"})

    pred = pred.merge(
        va_ids[["store_nbr", "item_nbr", "ds", "unique_id"]],
        on=["store_nbr", "item_nbr"],
        how="left",
    )
    pred = pred[pred["date"] == pred["ds"]].copy()
    pred = pred.merge(
        fcst[["unique_id", "ds", snaive_col, ets_col, theta_col]],
        on=["unique_id", "ds"],
        how="left",
    )

    pred["pred_snaive7"] = pred[snaive_col].fillna(pred["pred_naive"]).clip(lower=0)
    pred["pred_autoets"] = pred[ets_col].fillna(pred["pred_naive"]).clip(lower=0)
    pred["pred_autotheta"] = pred[theta_col].fillna(pred["pred_naive"]).clip(lower=0)

    baseline_scores = pd.DataFrame(
        {
            "model": ["Naive", "SeasonalNaive(7)", "AutoETS(7)", "AutoTheta(7)"],
            "NWRMSLE": [
                nwrmsle(pred["unit_sales"], pred["pred_naive"], pred["perishable"]),
                nwrmsle(pred["unit_sales"], pred["pred_snaive7"], pred["perishable"]),
                nwrmsle(pred["unit_sales"], pred["pred_autoets"], pred["perishable"]),
                nwrmsle(pred["unit_sales"], pred["pred_autotheta"], pred["perishable"]),
            ],
            "WMAPE": [
                wmape(pred["unit_sales"], pred["pred_naive"]),
                wmape(pred["unit_sales"], pred["pred_snaive7"]),
                wmape(pred["unit_sales"], pred["pred_autoets"]),
                wmape(pred["unit_sales"], pred["pred_autotheta"]),
            ],
            "BIAS": [
                bias_me(pred["unit_sales"], pred["pred_naive"]),
                bias_me(pred["unit_sales"], pred["pred_snaive7"]),
                bias_me(pred["unit_sales"], pred["pred_autoets"]),
                bias_me(pred["unit_sales"], pred["pred_autotheta"]),
            ],
        }
    ).sort_values("NWRMSLE").reset_index(drop=True)

    del tr_sf, sf_train, fcst, pred
    gc.collect()

    return baseline_scores
