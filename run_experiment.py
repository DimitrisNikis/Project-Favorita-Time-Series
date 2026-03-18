"""
Favorita Sales Forecasting — полный pipeline.

Usage:
    python run_experiment.py                     # полный запуск
    python run_experiment.py --skip-fe           # загрузить фичи из parquet
    python run_experiment.py --skip-baselines    # пропустить бейзлайны
"""

import argparse
import gc

import numpy as np
import pandas as pd

from src import config
from src.data.load import load_all_csvs
from src.data.prepare import prepare_side_tables, prepare_train_test
from src.features.dense_grid import attach_target_and_promo, make_dense_grid
from src.features.lags import add_point_lags
from src.features.pipeline import build_features_for_dates
from src.metrics import bias_me, nwrmsle, wmape
from src.models.catboost_model import (
    find_best_threshold,
    get_feature_list,
    make_submission,
    prep_xy,
    train_catboost,
)
from src.models.nn_model import (
    build_nn_seq,
    ensemble_search,
    fit_label_encoders,
    make_seq_dataset,
    predict_nn,
    train_nn,
)


def main():
    parser = argparse.ArgumentParser(description="Favorita Sales Forecasting Pipeline")
    parser.add_argument("--skip-fe", action="store_true",
                        help="Загрузить кэшированные features из parquet")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Пропустить статистические бейзлайны")
    args = parser.parse_args()

    # ── 1. Загрузка данных ───────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading data")
    print("=" * 60)
    data = load_all_csvs(config.CSV_DIR)
    print(f"  train: {data['train'].shape}, test: {data['test'].shape}")

    # ── 2. Подготовка ────────────────────────────────────────
    print("\nStep 2: Preparing data")
    train_work, test_work = prepare_train_test(
        data["train"], data["test"], data["items"]
    )
    items_w, stores_w, oil_w = prepare_side_tables(
        data["items"], data["stores"], data["oil"], train_work, test_work
    )
    del data
    gc.collect()
    print(f"  train_work: {train_work.shape}, test_work: {test_work.shape}")

    # ── 3. Feature engineering ───────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Feature Engineering")
    print("=" * 60)

    if args.skip_fe:
        print("  Loading pre-computed features from parquet...")
        train_dense_fe = pd.read_parquet(f"{config.SAVE_DIR}/train_fe_v2.parquet")
        valid_dense_fe = pd.read_parquet(f"{config.SAVE_DIR}/valid_fe_v2.parquet")
        test_fe = pd.read_parquet(f"{config.SAVE_DIR}/test_fe_v2.parquet")
    else:
        stores_u = np.sort(test_work["store_nbr"].unique()).astype("int16")
        items_u = np.sort(test_work["item_nbr"].unique()).astype("int32")

        train_dense_start = config.CV_TRAIN_END - pd.Timedelta(
            days=config.TRAIN_DENSE_DAYS - 1
        )
        train_grid = make_dense_grid(
            train_dense_start, config.CV_TRAIN_END, stores_u, items_u
        )
        valid_grid = make_dense_grid(
            config.CV_VALID_START, config.CV_VALID_END, stores_u, items_u
        )

        obs_train = train_work[
            (train_work["date"] >= train_dense_start)
            & (train_work["date"] <= config.CV_TRAIN_END)
        ]
        obs_valid = train_work[
            (train_work["date"] >= config.CV_VALID_START)
            & (train_work["date"] <= config.CV_VALID_END)
        ]

        train_dense = attach_target_and_promo(train_grid, obs_train)
        valid_dense = attach_target_and_promo(valid_grid, obs_valid)

        history = train_work[train_work["date"] <= config.CV_TRAIN_END].copy()
        train_dates = sorted(train_dense["date"].unique())
        valid_dates = sorted(valid_dense["date"].unique())
        test_dates = sorted(test_work["date"].unique())

        print(f"\nBuilding train features (gap=16)...")
        train_dense_fe = build_features_for_dates(
            train_dense, history, train_dates, items_w, stores_w, oil_w, gap=16
        )

        print("Building valid features (gap=0)...")
        valid_dense_fe = build_features_for_dates(
            valid_dense, history, valid_dates, items_w, stores_w, oil_w, gap=0
        )

        history_full = train_work.copy()
        print("Building test features (gap=0)...")
        test_fe = build_features_for_dates(
            test_work, history_full, test_dates, items_w, stores_w, oil_w,
            gap=0, is_test=True,
        )

    # Лаги
    history_for_lags = train_work[train_work["date"] <= config.CV_TRAIN_END][
        ["store_nbr", "item_nbr", "date", "unit_sales"]
    ]
    train_dense_fe = add_point_lags(train_dense_fe, history_for_lags)
    valid_dense_fe = add_point_lags(valid_dense_fe, history_for_lags)
    test_fe = add_point_lags(
        test_fe, train_work[["store_nbr", "item_nbr", "date", "unit_sales"]]
    )
    del history_for_lags
    gc.collect()
    print(f"  Shapes: train={train_dense_fe.shape}, valid={valid_dense_fe.shape}, test={test_fe.shape}")

    # ── 4. Baselines ─────────────────────────────────────────
    baseline_scores = None
    if not args.skip_baselines:
        print("\n" + "=" * 60)
        print("Step 4: Statistical Baselines")
        print("=" * 60)
        from src.baselines import run_baselines

        baseline_scores = run_baselines(train_dense_fe, valid_dense_fe)
        print(baseline_scores.to_string(index=False))

    # ── 5. CatBoost ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: CatBoost")
    print("=" * 60)
    features = get_feature_list(train_dense_fe, valid_dense_fe)
    X_tr, y_tr, w_tr = prep_xy(train_dense_fe, features)
    X_va, y_va, w_va = prep_xy(valid_dense_fe, features)
    print(f"  Features: {len(features)}, X_tr: {X_tr.shape}, X_va: {X_va.shape}")

    model = train_catboost(X_tr, y_tr, w_tr, X_va, y_va, w_va)
    pred_va = np.expm1(model.predict(X_va)).clip(0)

    score_cb = nwrmsle(valid_dense_fe["unit_sales"], pred_va, valid_dense_fe["perishable"])
    print(f"\n  CatBoost CV NWRMSLE: {score_cb:.6f}")

    # Threshold tuning
    best_t, thr_tbl = find_best_threshold(
        pred_va,
        valid_dense_fe["unit_sales"].values,
        valid_dense_fe["perishable"].values,
    )
    print(f"  Best threshold: {best_t}")

    # CatBoost submission
    import os
    drive_results = os.path.join(config.BASE_DIR, "results")
    local_results = "results"
    os.makedirs(drive_results, exist_ok=True)
    os.makedirs(local_results, exist_ok=True)

    for path in [
        os.path.join(drive_results, "submission_catboost_v3.csv.gz"),
        os.path.join(local_results, "submission_catboost_v3.csv.gz"),
    ]:
        make_submission(model, test_fe, features, config.CAT_COLS, best_t, path)

    # CatBoost test predictions (needed for ensemble later)
    from src.models.catboost_model import prep_xy as _prep_xy
    X_test_cb = test_fe[features].copy()
    for c in config.CAT_COLS:
        if c in X_test_cb.columns:
            X_test_cb[c] = X_test_cb[c].astype(str).fillna("UNK")
    for c in features:
        if c not in config.CAT_COLS:
            X_test_cb[c] = pd.to_numeric(X_test_cb[c], errors="coerce").fillna(0).astype("float32")
    pred_test_cb = np.expm1(model.predict(X_test_cb)).clip(0)

    # ── 6. Neural Network ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: Neural Network")
    print("=" * 60)

    le_dict, emb_sizes = fit_label_encoders(train_dense_fe, valid_dense_fe, test_fe)
    static_cols_nn = [c for c in train_dense_fe.columns
                      if c not in config.EXCLUDE_NN and c in valid_dense_fe.columns]
    print(f"  Static features: {len(static_cols_nn)}")
    print(f"  Embedding sizes: {emb_sizes}")

    print("\n  Preparing sequence datasets...")
    X_tr_st, X_tr_emb, promo_tr, y_tr_seq, w_tr_seq = make_seq_dataset(
        train_dense_fe, le_dict, static_cols_nn
    )
    X_va_st, X_va_emb, promo_va, y_va_seq, w_va_seq = make_seq_dataset(
        valid_dense_fe, le_dict, static_cols_nn
    )
    print(f"  X_tr_st: {X_tr_st.shape}, promo_tr: {promo_tr.shape}, y_tr_seq: {y_tr_seq.shape}")
    print(f"  X_va_st: {X_va_st.shape}, promo_va: {promo_va.shape}, y_va_seq: {y_va_seq.shape}")

    model_nn = build_nn_seq(len(static_cols_nn), emb_sizes, config.H)
    model_nn.summary()

    print("\n  Training NN...")
    model_nn, history_nn = train_nn(
        model_nn,
        X_tr_st, X_tr_emb, promo_tr, y_tr_seq, w_tr_seq,
        X_va_st, X_va_emb, promo_va, y_va_seq, w_va_seq,
    )

    pred_nn_va = predict_nn(model_nn, X_va_st, X_va_emb, promo_va, valid_dense_fe)
    score_nn = nwrmsle(valid_dense_fe["unit_sales"], pred_nn_va, valid_dense_fe["perishable"])
    print(f"\n  NN (sequence + Conv1D) CV NWRMSLE: {score_nn:.6f}")

    # ── 7. Ensemble CatBoost + NN ───────────────────────────
    print("\n" + "=" * 60)
    print("Step 7: Ensemble Search")
    print("=" * 60)

    best_alpha, best_blend_score = ensemble_search(
        pred_va, pred_nn_va,
        valid_dense_fe["unit_sales"].values,
        valid_dense_fe["perishable"].values,
        best_t,
    )

    pred_blend_va = best_alpha * pred_va + (1 - best_alpha) * pred_nn_va
    pred_blend_va = np.where(pred_blend_va < best_t, 0.0, pred_blend_va)

    # Ensemble submission
    X_te_st, X_te_emb, promo_te = make_seq_dataset(
        test_fe, le_dict, static_cols_nn, is_test=True
    )
    pred_nn_test = predict_nn(model_nn, X_te_st, X_te_emb, promo_te, test_fe)

    pred_blend_test = best_alpha * pred_test_cb + (1 - best_alpha) * pred_nn_test
    pred_blend_test = np.where(pred_blend_test < best_t, 0.0, pred_blend_test).astype("float32")

    sub_blend = test_fe[["id"]].copy()
    sub_blend["unit_sales"] = pred_blend_test
    for path in [drive_results, local_results]:
        sub_blend.to_csv(
            os.path.join(path, "submission_ensemble.csv.gz"),
            index=False, compression="gzip",
        )
    print("  Saved: submission_ensemble.csv.gz")

    # ── 8. Итоговая таблица ──────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULTS (all models)")
    print("=" * 60)

    pred_va_thr = pred_va.copy()
    pred_va_thr[pred_va_thr < best_t] = 0.0

    cb_row = pd.DataFrame([{
        "model": "CatBoost v3 (winner features + gap=16)",
        "NWRMSLE": nwrmsle(valid_dense_fe["unit_sales"], pred_va_thr, valid_dense_fe["perishable"]),
        "WMAPE": wmape(valid_dense_fe["unit_sales"], pred_va_thr),
        "BIAS": bias_me(valid_dense_fe["unit_sales"], pred_va_thr),
    }])
    nn_row = pd.DataFrame([{
        "model": "NN (sequence + Conv1D promo)",
        "NWRMSLE": score_nn,
        "WMAPE": wmape(valid_dense_fe["unit_sales"], pred_nn_va),
        "BIAS": bias_me(valid_dense_fe["unit_sales"], pred_nn_va),
    }])
    ens_row = pd.DataFrame([{
        "model": f"Ensemble CatBoost+NN (alpha={best_alpha:.1f})",
        "NWRMSLE": best_blend_score,
        "WMAPE": wmape(valid_dense_fe["unit_sales"], pred_blend_va),
        "BIAS": bias_me(valid_dense_fe["unit_sales"], pred_blend_va),
    }])

    parts = [cb_row, nn_row, ens_row]
    if baseline_scores is not None:
        parts.insert(0, baseline_scores)

    all_scores = pd.concat(parts, ignore_index=True).sort_values("NWRMSLE").reset_index(drop=True)
    print(all_scores.to_string(index=False))

    for path in [drive_results, local_results]:
        all_scores.to_csv(os.path.join(path, "scores.csv"), index=False)
    print(f"\n  Results saved to Drive and local")

    gc.collect()
    print("\nDone!")


if __name__ == "__main__":
    main()
