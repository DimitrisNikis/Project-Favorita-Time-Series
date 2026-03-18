"""Загрузка CSV-файлов конкурса."""

import os

import pandas as pd


def load_all_csvs(csv_dir):
    """Загружает все CSV и возвращает dict[str, DataFrame]."""
    data = {}
    data["train"] = pd.read_csv(os.path.join(csv_dir, "train.csv"), low_memory=False)
    data["test"] = pd.read_csv(os.path.join(csv_dir, "test.csv"))
    data["items"] = pd.read_csv(os.path.join(csv_dir, "items.csv"))
    data["stores"] = pd.read_csv(os.path.join(csv_dir, "stores.csv"))
    data["oil"] = pd.read_csv(os.path.join(csv_dir, "oil.csv"))
    data["holidays"] = pd.read_csv(os.path.join(csv_dir, "holidays_events.csv"))
    data["transactions"] = pd.read_csv(os.path.join(csv_dir, "transactions.csv"))
    
    for key in ["train", "test", "oil", "holidays", "transactions"]:
        if "date" in data[key].columns:
            data[key]["date"] = pd.to_datetime(data[key]["date"])

    return data
