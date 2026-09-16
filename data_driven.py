from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from common import (BASE_FEATURES, log_to_physical, metrics, prepare_comparison_dataframe,
                    prepare_features, save_predictions_plot, set_seed, sklearn_cv,
                    split_indices, target_to_log)


def models(seed=42):
    return {
        "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.05, objective="reg:squarederror"),
        "SVM": SVR(kernel="rbf", C=10.0, epsilon=0.02, gamma="scale"),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=25, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=2),
        "Extra Trees": ExtraTreesRegressor(n_estimators=150, max_depth=25, random_state=42),
    }


def build_ann(input_dim: int, seed: int = 42):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss="mse", metrics=["mae"])
    return model


def fit_ann(X, y, train, val, seed=42, epochs=500):
    import tensorflow as tf
    model = build_ann(X.shape[1], seed)
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, min_delta=1e-4,
                                             restore_best_weights=True)
    model.fit(X[train], y[train], validation_data=(X[val], y[val]), epochs=epochs,
              batch_size=32, verbose=1, callbacks=[stop])
    return model


def run(df: pd.DataFrame, output_dir: str | Path, seed=42, epochs=500,
        with_ann=True, with_cv=True) -> pd.DataFrame:
    set_seed(seed)
    out = Path(output_dir) / "data_driven"; out.mkdir(parents=True, exist_ok=True)
    data = prepare_comparison_dataframe(df)
    X, _, _ = prepare_features(data, BASE_FEATURES)
    y = target_to_log(data["Wear rate"].to_numpy(float))
    train, val, test = split_indices(len(data), 42)

    if with_cv:
        sklearn_cv(models(seed), X[train], y[train]).to_csv(out / "cross_validation.csv", index=False)

    rows = []
    for name, model in models(seed).items():
        model.fit(X[train], y[train])
        pred_log = model.predict(X[test]); pred = log_to_physical(pred_log)
        true = data["Wear rate"].to_numpy(float)[test]
        rows.append({"model": name, **metrics(true, pred, y[test], pred_log)})
        save_predictions_plot(true, pred, name, out / f"pred_{name.replace(' ', '_')}.png")

    if with_ann:
        ann = fit_ann(X, y, train, val, seed, epochs)
        pred_log = ann.predict(X[test], verbose=0).ravel(); pred = log_to_physical(pred_log)
        true = data["Wear rate"].to_numpy(float)[test]
        rows.append({"model": "ANN", **metrics(true, pred, y[test], pred_log)})
        save_predictions_plot(true, pred, "ANN", out / "pred_ANN.png")

    table = pd.DataFrame(rows); table.to_csv(out / "metrics.csv", index=False)
    return table
