"""Метрики оценки качества прогнозов."""

import numpy as np


def nwrmsle(y_true, y_pred, perishable):
    """Normalized Weighted RMSLE (основная метрика соревнования)."""
    y_true = np.clip(np.asarray(y_true, float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    w = np.where(np.asarray(perishable).astype(int) == 1, 1.25, 1.0)
    e = np.log1p(y_pred) - np.log1p(y_true)
    return np.sqrt((w * e**2).sum() / w.sum())


def wmape(y_true, y_pred, eps=1e-9):
    """Weighted Mean Absolute Percentage Error."""
    y_true = np.clip(np.asarray(y_true, float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    return np.abs(y_true - y_pred).sum() / (np.abs(y_true).sum() + eps)


def bias_me(y_true, y_pred):
    """Mean bias."""
    y_true = np.clip(np.asarray(y_true, float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    return (y_pred - y_true).mean()
