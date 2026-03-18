"""
Нейросеть: sequence + Conv1D promo trick.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src import config
from src.metrics import nwrmsle


def fit_label_encoders(train_df, valid_df, test_df):
    """Обучает LabelEncoder на объединении train+valid+test для каждой EMB_COL.

    Returns:
        le_dict:   dict {col_name: обученный LabelEncoder}
        emb_sizes: dict {col_name: (n_categories, embedding_dim)}
    """
    le_dict = {}
    for c in config.EMB_COLS:
        le = LabelEncoder()
        all_vals = pd.concat([
            train_df[c].astype(str),
            valid_df[c].astype(str),
            test_df[c].astype(str),
        ])
        le.fit(all_vals)
        le_dict[c] = le

    # Размерность эмбеддинга = min(50, (n_classes + 1) // 2)
    emb_sizes = {
        c: (len(le_dict[c].classes_), min(50, (len(le_dict[c].classes_) + 1) // 2))
        for c in config.EMB_COLS
    }
    return le_dict, emb_sizes


def _get_static_cols(train_df, valid_df):
    """Определяет числовые статические колонки для NN (без динамических, категориальных, мета)."""
    return [c for c in train_df.columns
            if c not in config.EXCLUDE_NN and c in valid_df.columns]


def make_seq_dataset(df, le_dict, static_cols, is_test=False):
    """Преобразует DataFrame в sequence-формат для NN.

    Для valid/test (16 дней): прямой reshape в (n_series, 16).
    Для train (50 дней): нарезка на непересекающиеся 16-дневные окна.

    Returns:
        X_static: np.array (n_samples, n_static_features)
        X_emb:    dict {col: np.array (n_samples,)}
        promo:    np.array (n_samples, H)
        y:        np.array (n_samples, H)  — только если не is_test
        w:        np.array (n_samples,)    — только если не is_test
    """
    H = config.H
    df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)
    n_series_raw = df[["store_nbr", "item_nbr"]].drop_duplicates().shape[0]
    n_days = len(df) // n_series_raw

    if n_days == H:
        windows = [df]
    else:
        all_dates = sorted(df["date"].unique())
        windows = []
        for start in range(0, len(all_dates) - H + 1, H):
            w_dates = all_dates[start:start + H]
            windows.append(df[df["date"].isin(w_dates)])
        print(f"  Train: {n_days} дней -> {len(windows)} окон по {H} дней")

    X_static_list, X_emb_list = [], {c: [] for c in config.EMB_COLS}
    promo_list, y_list, w_list = [], [], []

    for wdf in windows:
        wdf = wdf.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)
        n_ser = len(wdf) // H
        assert len(wdf) % H == 0

        first_rows = wdf.iloc[::H]

        X_s = first_rows[static_cols].copy()
        for c in static_cols:
            X_s[c] = pd.to_numeric(X_s[c], errors="coerce").fillna(0)
        X_static_list.append(X_s.values.astype("float32"))

        for c in config.EMB_COLS:
            X_emb_list[c].append(
                le_dict[c].transform(first_rows[c].astype(str)).astype("int32")
            )

        promo_list.append(
            wdf["onpromotion"].values.reshape(n_ser, H).astype("float32")
        )

        if not is_test:
            y_list.append(
                np.log1p(
                    wdf["unit_sales"].clip(lower=0).values
                    .reshape(n_ser, H).astype("float32")
                )
            )
            perishable = wdf["perishable"].values.reshape(n_ser, H)[:, 0]
            w_list.append(
                np.where(perishable == 1, 1.25, 1.0).astype("float32")
            )

    X_static = np.concatenate(X_static_list, axis=0)
    X_emb = {c: np.concatenate(X_emb_list[c], axis=0) for c in config.EMB_COLS}
    promo = np.concatenate(promo_list, axis=0)

    if is_test:
        return X_static, X_emb, promo

    y = np.concatenate(y_list, axis=0)
    w = np.concatenate(w_list, axis=0)
    return X_static, X_emb, promo, y, w


def build_nn_seq(n_static, emb_sizes, horizon=16):
    """Строит Keras-модель: embeddings + dense + Conv1D promo trick (8-е место)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    static_input = keras.Input(shape=(n_static,), name="numeric")
    x = layers.Dense(512, activation="relu",
                     kernel_initializer="he_normal")(static_input)
    x = layers.Dropout(0.2)(x)

    emb_inputs, emb_outputs = [], []
    for col, (n_cats, emb_dim) in emb_sizes.items():
        inp = keras.Input(shape=(1,), name=col)
        emb = layers.Embedding(n_cats, emb_dim, name=f"emb_{col}")(inp)
        emb = layers.Flatten()(emb)
        emb_inputs.append(inp)
        emb_outputs.append(emb)

    x = layers.Concatenate()([x] + emb_outputs)
    x = layers.Dense(256, activation="relu",
                     kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)

    x = layers.Dense(horizon, activation="relu")(x)

    promo_input = keras.Input(shape=(horizon,), name="promo")

    y = layers.Multiply()([x, promo_input])
    x_r = layers.Reshape((horizon, 1))(x)
    y_r = layers.Reshape((horizon, 1))(y)
    z_r = layers.Reshape((horizon, 1))(promo_input)
    combined = layers.Concatenate(axis=-1)([x_r, y_r, z_r])
    out = layers.Conv1D(1, 1, activation="linear")(combined)
    output = layers.Reshape((horizon,))(out)

    output = layers.Activation("sigmoid")(output)
    output = layers.Lambda(lambda t: t * 10.0)(output)

    model = keras.Model(
        inputs=[static_input] + emb_inputs + [promo_input],
        outputs=output,
    )
    return model


def train_nn(model, X_tr_st, X_tr_emb, promo_tr, y_tr, w_tr,
             X_va_st, X_va_emb, promo_va, y_va, w_va):
    """Компилирует и обучает NN-модель. Возвращает (model, history)."""
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.NN_LR),
        loss="mse",
    )

    train_inputs = ([X_tr_st]
                    + [X_tr_emb[c] for c in config.EMB_COLS]
                    + [promo_tr])
    valid_inputs = ([X_va_st]
                    + [X_va_emb[c] for c in config.EMB_COLS]
                    + [promo_va])

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=config.NN_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=4, min_lr=1e-5, verbose=1,
        ),
    ]

    history = model.fit(
        train_inputs, y_tr,
        sample_weight=w_tr,
        validation_data=(valid_inputs, y_va, w_va),
        epochs=config.NN_EPOCHS,
        batch_size=config.NN_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def predict_nn(model, X_st, X_emb, promo, df):
    """Генерирует предсказания и выравнивает порядок с исходным DataFrame.

    Returns:
        pred: np.array shape (len(df),) — предсказания в порядке исходного df
    """
    inputs = ([X_st]
              + [X_emb[c] for c in config.EMB_COLS]
              + [promo])

    pred_seq = np.expm1(
        np.clip(model.predict(inputs, batch_size=4096), 0, 10)
    ).clip(0).flatten()

    sorted_idx = df.sort_values(
        ["store_nbr", "item_nbr", "date"]
    ).index
    pred = np.empty(len(df))
    pred[sorted_idx] = pred_seq
    return pred


def ensemble_search(pred_catboost, pred_nn, y_true, perishable, threshold,
                    alpha_min=0.0, alpha_max=1.0, alpha_step=0.1):
    """Grid search по alpha для ансамбля CatBoost + NN."""
    best_alpha, best_score = 0.5, float("inf")

    for alpha in np.arange(alpha_min, alpha_max + alpha_step, alpha_step):
        blend = alpha * pred_catboost + (1 - alpha) * pred_nn
        blend = np.where(blend < threshold, 0.0, blend)
        s = nwrmsle(y_true, blend, perishable)
        print(f"  alpha={alpha:.1f} (CatBoost) + {1-alpha:.1f} (NN) -> NWRMSLE: {s:.6f}")
        if s < best_score:
            best_score, best_alpha = s, alpha

    print(f"\nЛучший ансамбль: alpha={best_alpha:.1f} -> NWRMSLE {best_score:.6f}")
    return best_alpha, best_score
