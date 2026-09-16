from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common import (
    BASE_FEATURES, SCALE_FACTOR, log_to_physical, metrics, prepare_features,
    prepare_pgnn_dataframe, save_predictions_plot, set_seed, split_indices,
    target_to_log,
)
from data_driven import build_ann

PGNN_FEATURES = BASE_FEATURES + ["Wear volume (mm^3)"]


def _fit_ann(X, y, train, val, seed, epochs):
    import tensorflow as tf

    model = build_ann(X.shape[1], seed)
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, min_delta=1e-4, restore_best_weights=True
    )
    model.fit(
        X[train], y[train], validation_data=(X[val], y[val]),
        epochs=epochs, batch_size=32, verbose=1, callbacks=[stop]
    )
    return model


def _fit_pgnn(X, y, V, F, L, mask, train, val, seed, epochs):
    import tensorflow as tf

    model = build_ann(X.shape[1], seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    data_mean = tf.Variable(1.0, trainable=False, dtype=tf.float32)
    phys_mean = tf.Variable(1.0, trainable=False, dtype=tf.float32)
    rng = np.random.default_rng(seed)
    best_weights, best_val, wait = None, np.inf, 0
    eps = tf.constant(1e-8, tf.float32)

    @tf.function
    def train_batch(xb, yb, vb, fb, lb, mb):
        with tf.GradientTape() as tape:
            pred_log = tf.squeeze(model(xb, training=True), axis=-1)
            l_data = tf.reduce_mean(tf.square(yb - pred_log))

            K = tf.pow(tf.constant(10.0, tf.float32), pred_log) / tf.constant(SCALE_FACTOR, tf.float32)
            residual = (K * fb * lb - vb) * mb
            denom = tf.maximum(tf.reduce_sum(mb), 1.0)
            l_phys = tf.reduce_sum(tf.square(residual)) / denom

            data_mean.assign(0.99 * data_mean + 0.01 * l_data)
            phys_mean.assign(0.99 * phys_mean + 0.01 * l_phys)
            loss = l_data / (data_mean + eps) + l_phys / (phys_mean + eps)

            if model.losses:
                loss += tf.add_n(model.losses)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train = np.asarray(train)
    for _ in range(epochs):
        order = rng.permutation(train)
        for start in range(0, len(train), 32):
            b = order[start:start + 32]
            train_batch(
                tf.convert_to_tensor(X[b], tf.float32),
                tf.convert_to_tensor(y[b], tf.float32),
                tf.convert_to_tensor(V[b], tf.float32),
                tf.convert_to_tensor(F[b], tf.float32),
                tf.convert_to_tensor(L[b], tf.float32),
                tf.convert_to_tensor(mask[b], tf.float32),
            )

        val_pred = model.predict(X[val], verbose=0).ravel()
        val_loss = float(np.mean((y[val] - val_pred) ** 2))
        if val_loss < best_val - 1e-4:
            best_val, best_weights, wait = val_loss, model.get_weights(), 0
        else:
            wait += 1
            if wait >= 30:
                break

    if best_weights is not None:
        model.set_weights(best_weights)
    return model


def _configuration(data, label, out, seed, epochs):
    X, _, _ = prepare_features(data, PGNN_FEATURES)
    y = target_to_log(data["Wear rate"].to_numpy(float))
    train, val, test = split_indices(len(data), 42)

    Vraw = pd.to_numeric(data["Volume (mm^3)"], errors="coerce").to_numpy(float)
    Fraw = pd.to_numeric(data["Load (N)"], errors="coerce").to_numpy(float)
    Lraw = pd.to_numeric(data["Sliding distance (m)"], errors="coerce").to_numpy(float)
    mask = np.isfinite(Vraw) & np.isfinite(Fraw) & np.isfinite(Lraw)

    V = np.nan_to_num(Vraw, nan=0.0).astype(np.float32)
    F = np.nan_to_num(Fraw, nan=0.0).astype(np.float32)
    L = np.nan_to_num(Lraw, nan=0.0).astype(np.float32)
    m = mask.astype(np.float32)
    true = data["Wear rate"].to_numpy(float)[test]

    ann = _fit_ann(X, y, train, val, seed, epochs)
    ann_log = ann.predict(X[test], verbose=0).ravel()
    ann_pred = log_to_physical(ann_log)
    rows = [{"configuration": label, "model": "ANN", **metrics(true, ann_pred, y[test], ann_log)}]
    save_predictions_plot(true, ann_pred, f"ANN - {label}", out / f"pred_ANN_{label}.png")

    pgnn = _fit_pgnn(X, y, V, F, L, m, train, val, seed, epochs)
    pgnn_log = pgnn.predict(X[test], verbose=0).ravel()
    pgnn_pred = log_to_physical(pgnn_log)
    rows.append({"configuration": label, "model": "PGNN", **metrics(true, pgnn_pred, y[test], pgnn_log)})
    save_predictions_plot(true, pgnn_pred, f"PGNN - {label}", out / f"pred_PGNN_{label}.png")
    return rows


def run(df: pd.DataFrame, output_dir: str | Path, seed=42, epochs=500) -> pd.DataFrame:
    set_seed(seed)
    out = Path(output_dir) / "pgnn"
    out.mkdir(parents=True, exist_ok=True)

    data = prepare_pgnn_dataframe(df)
    if "Volume (mm^3)" not in data.columns:
        raise KeyError("PGNN requires measured 'Volume (mm^3)'.")

    rows = _configuration(data, "full", out, seed, epochs)
    complete = data.dropna(
        subset=["Volume (mm^3)", "Load (N)", "Sliding distance (m)"]
    ).reset_index(drop=True)

    if len(complete) >= 20:
        rows.extend(_configuration(complete, "reduced", out, seed, epochs))

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False)
    return table
