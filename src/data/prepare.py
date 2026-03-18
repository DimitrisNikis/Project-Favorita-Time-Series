"""Подготовка рабочих датафреймов: фильтрация, типы, вспомогательные таблицы."""

import pandas as pd

from src.config import HISTORY_START


def prepare_train_test(train, test, items):
    """Фильтрует train по HISTORY_START, конвертирует типы, добавляет class."""
    train_work = train.loc[
        train["date"] >= HISTORY_START,
        ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
    ].copy()
    test_work = test[["id", "date", "store_nbr", "item_nbr", "onpromotion"]].copy()

    promo_map = {True: 1, False: 0, "True": 1, "False": 0}
    train_work["onpromotion"] = (
        train_work["onpromotion"].map(promo_map).fillna(0).astype("int8")
    )
    test_work["onpromotion"] = (
        test_work["onpromotion"].map(promo_map).fillna(0).astype("int8")
    )

    train_work["unit_sales"] = train_work["unit_sales"].astype("float32").clip(lower=0)
    train_work["store_nbr"] = train_work["store_nbr"].astype("int16")
    test_work["store_nbr"] = test_work["store_nbr"].astype("int16")
    train_work["item_nbr"] = train_work["item_nbr"].astype("int32")
    test_work["item_nbr"] = test_work["item_nbr"].astype("int32")

    train_work = train_work.merge(
        items[["item_nbr", "class"]].astype({"item_nbr": "int32", "class": "int32"}),
        on="item_nbr",
        how="left",
    )
    return train_work, test_work


def prepare_side_tables(items, stores, oil, train_work, test_work):
    """Готовит items_w, stores_w, oil_w с правильными типами."""
    items_w = items[["item_nbr", "family", "class", "perishable"]].copy()
    items_w["item_nbr"] = items_w["item_nbr"].astype("int32")
    items_w["class"] = items_w["class"].astype("int32")
    items_w["perishable"] = items_w["perishable"].astype("int8")
    items_w["family"] = items_w["family"].astype("category")

    stores_w = stores[["store_nbr", "city", "state", "type", "cluster"]].copy()
    stores_w["store_nbr"] = stores_w["store_nbr"].astype("int16")
    stores_w["cluster"] = stores_w["cluster"].astype("int16")
    for c in ["city", "state", "type"]:
        stores_w[c] = stores_w[c].astype("category")

    oil_src = oil[["date", "dcoilwtico"]].copy()
    oil_src["date"] = pd.to_datetime(oil_src["date"])
    min_d = min(train_work["date"].min(), test_work["date"].min())
    max_d = max(train_work["date"].max(), test_work["date"].max())
    all_days = pd.DataFrame({"date": pd.date_range(min_d, max_d, freq="D")})
    oil_w = all_days.merge(oil_src, on="date", how="left")
    oil_w["dcoilwtico"] = oil_w["dcoilwtico"].ffill().bfill().astype("float32")

    return items_w, stores_w, oil_w
