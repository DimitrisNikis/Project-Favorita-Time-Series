"""CatBoost: подготовка данных, обучение, threshold tuning, submission."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.config import CAT_COLS, CATBOOST_PARAMS, EXCLUDE_COLS
from src.metrics import nwrmsle


def get_feature_list(train_df, valid_df, exclude_cols=None):
    """Определяет список признаков (общие колонки минус исключённые)."""
    if exclude_cols is None:
        exclude_cols = EXCLUDE_COLS
    return [
        c for c in train_df.columns
        if c not in exclude_cols and c in valid_df.columns
    ]


def prep_xy(df, features, cat_features=None):
    """Готовит X, y (log1p), w (1.25 для perishable) для обучения."""
    if cat_features is None:
        cat_features = CAT_COLS
    X = df[features].copy()
    for c in cat_features:
        X[c] = X[c].astype(str).fillna("UNK")
    for c in features:
        if c not in cat_features:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype("float32")
    y = np.log1p(np.clip(df["unit_sales"].values.astype(float), 0, None))
    w = np.where(df["perishable"].values.astype(int) == 1, 1.25, 1.0)
    return X, y, w


def prep_X(df, features, cat_features=None):
    """Готовит X для предсказания (без target)."""
    if cat_features is None:
        cat_features = CAT_COLS
    X = (
        df[features].copy()
        if all(c in df.columns for c in features)
        else df.reindex(columns=features).copy()
    )
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("UNK")
        else:
            X[c] = "UNK"
    for c in features:
        if c not in cat_features:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype("float32")
    return X


def train_catboost(X_tr, y_tr, w_tr, X_va, y_va, w_va, cat_cols=None, params=None):
    """Обучает CatBoost и возвращает модель."""
    if cat_cols is None:
        cat_cols = CAT_COLS
    if params is None:
        params = CATBOOST_PARAMS

    train_pool = Pool(X_tr, y_tr, cat_features=cat_cols, weight=w_tr)
    valid_pool = Pool(X_va, y_va, cat_features=cat_cols, weight=w_va)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def find_best_threshold(pred_va, y_true, perishable,
                         grid_min=0.0, grid_max=2.0, grid_step=0.05):
    """Подбирает порог обнуления для минимизации NWRMSLE."""
    grid_t = np.arange(grid_min, grid_max + grid_step, grid_step)
    rows = []
    for t in grid_t:
        p = pred_va.copy()
        p[p < t] = 0.0
        rows.append((t, nwrmsle(y_true, p, perishable)))

    thr_tbl = (
        pd.DataFrame(rows, columns=["threshold", "NWRMSLE"])
        .sort_values("NWRMSLE")
        .reset_index(drop=True)
    )
    best_t = float(thr_tbl.loc[0, "threshold"])
    return best_t, thr_tbl


def make_submission(model, test_fe, features, cat_cols, threshold, output_path):
    """Генерирует submission CSV."""
    X_test = prep_X(test_fe, features, cat_cols)
    pred_test = np.expm1(model.predict(X_test)).clip(0)
    pred_test[pred_test < threshold] = 0.0

    sub = test_fe[["id"]].copy()
    sub["unit_sales"] = pred_test.astype("float32")

    print(f"Submission shape: {sub.shape}")
    print(f"Zero%: {round((sub['unit_sales'] == 0).mean() * 100, 1)}")

    sub.to_csv(output_path, index=False, compression="gzip")
    print(f"Saved: {output_path}")
    return sub
