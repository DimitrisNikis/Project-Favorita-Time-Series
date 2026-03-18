"""Распаковка 7z-архивов с данными через py7zr (без системных зависимостей)."""

import glob
import os

import py7zr


def extract_archives(raw_dir, csv_dir):
    """Распаковывает все .7z архивы из raw_dir в csv_dir."""
    os.makedirs(csv_dir, exist_ok=True)
    archives = sorted(glob.glob(os.path.join(raw_dir, "*.7z")))
    print(f"Найдено архивов: {len(archives)}")
    for f in archives:
        print(f"Распаковываю: {os.path.basename(f)}")
        with py7zr.SevenZipFile(f, mode="r") as z:
            z.extractall(path=csv_dir)
    print("Готово")
