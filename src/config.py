"""Центральная конфигурация: все константы, гиперпараметры и пути."""

import os

import pandas as pd


# ── Пути (Google Colab + Google Drive) ──────────────────────
BASE_DIR = "/content/drive/MyDrive/favorita"
RAW_DIR = os.path.join(BASE_DIR, "raw_7z")
CSV_DIR = os.path.join(BASE_DIR, "csv")
SAVE_DIR = os.path.join(BASE_DIR, "features_v2")

# ── Горизонт и даты ─────────────────────────────────────────
H = 16
START_DATE = "2017-01-01"
HISTORY_START = "2016-08-01"

CV_TRAIN_END = pd.Timestamp("2017-07-30")
CV_VALID_START = pd.Timestamp("2017-07-31")
CV_VALID_END = pd.Timestamp("2017-08-15")

# ── Feature engineering ──────────────────────────────────────
STAT_WINDOWS = [3, 7, 14, 30, 60, 140]
LAGS = [16, 23, 30, 37, 44]
TRAIN_DENSE_DAYS = 50

# ── Baselines ────────────────────────────────────────────────
MAX_SERIES = 30_000

# ── CatBoost ─────────────────────────────────────────────────
EXCLUDE_COLS = {"id", "date", "unit_sales", "unit_sales_raw", "city", "state"}
CAT_COLS = ["family", "type", "store_nbr", "item_nbr", "class", "cluster"]

CATBOOST_PARAMS = dict(
    loss_function="RMSE",
    eval_metric="RMSE",
    task_type="GPU",
    depth=8,
    learning_rate=0.03,
    n_estimators=2500,
    l2_leaf_reg=3,
    random_seed=42,
    early_stopping_rounds=200,
    verbose=200,
)

THRESHOLD_GRID_MIN = 0.0
THRESHOLD_GRID_MAX = 2.0
THRESHOLD_GRID_STEP = 0.05

# ── Neural Network ───────────────────────────────────────────
NN_EPOCHS = 50
NN_BATCH_SIZE = 2048
NN_LR = 3e-4
NN_PATIENCE = 7

EMB_COLS = ["family", "type", "store_nbr", "item_nbr", "class", "cluster"]
DYNAMIC_COLS = {"onpromotion", "dow", "day", "is_payday"}
EXCLUDE_NN = {"id", "date", "unit_sales", "unit_sales_raw",
              "city", "state", "perishable"} | set(EMB_COLS) | DYNAMIC_COLS
