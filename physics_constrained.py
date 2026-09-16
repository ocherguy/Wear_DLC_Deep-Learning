from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common import (BASE_FEATURES, log_to_physical, metrics, prepare_comparison_dataframe,
                    prepare_features, save_predictions_plot, set_seed, sklearn_cv,
                    split_indices, target_to_log)
from data_driven import fit_ann, models


def _target(data: pd.DataFrame, formulation: str):
    if formulation == "dV": return data["dV"].to_numpy(float)
    if formulation == "k": return data["k"].to_numpy(float)
    if formulation == "alpha": return data["alpha"].to_numpy(float)
    raise ValueError("formulation must be dV, k or alpha")


def _reconstruct(z, data: pd.DataFrame, idx, formulation: str):
    z = np.asarray(z).reshape(-1)
    F = data["Load (N)"].to_numpy(float)[idx]
    H = data["Film hardness (GPa)"].to_numpy(float)[idx]
    mu = data["Friction coefficient"].to_numpy(float)[idx]
    if formulation == "dV": return z / F
    if formulation == "k": return z / H
    return z * mu / H


def run(df: pd.DataFrame, output_dir: str | Path, formulations=("dV", "k", "alpha"),
        seed=42, epochs=500, with_ann=True, with_cv=True) -> pd.DataFrame:
    set_seed(seed)
    out = Path(output_dir) / "physics_constrained"; out.mkdir(parents=True, exist_ok=True)
    data = prepare_comparison_dataframe(df)
    X, _, _ = prepare_features(data, BASE_FEATURES)
    train, val, test = split_indices(len(data), 42)
    true_wear = data["Wear rate"].to_numpy(float)
    rows = []

    for formulation in formulations:
        target_phys = _target(data, formulation)
        y = target_to_log(target_phys)
        target_dir = out / formulation; target_dir.mkdir(exist_ok=True)

        if with_cv:
            cv = sklearn_cv(models(seed), X[train], y[train])
            cv.insert(0, "formulation", formulation)
            cv.to_csv(target_dir / "cross_validation.csv", index=False)

        trained = models(seed)
        for name, model in trained.items():
            model.fit(X[train], y[train])
            pred_log = model.predict(X[test])
            pred_wear = _reconstruct(log_to_physical(pred_log), data, test, formulation)
            # Log metrics are evaluated in the learned target space, as in the source script.
            rows.append({"formulation": formulation, "model": name,
                         **metrics(true_wear[test], pred_wear, y[test], pred_log)})
            save_predictions_plot(true_wear[test], pred_wear, f"{name} - {formulation}",
                                  target_dir / f"pred_{name.replace(' ', '_')}.png")

        if with_ann:
            ann = fit_ann(X, y, train, val, seed, epochs)
            pred_log = ann.predict(X[test], verbose=0).ravel()
            pred_wear = _reconstruct(log_to_physical(pred_log), data, test, formulation)
            rows.append({"formulation": formulation, "model": "ANN",
                         **metrics(true_wear[test], pred_wear, y[test], pred_log)})
            save_predictions_plot(true_wear[test], pred_wear, f"ANN - {formulation}",
                                  target_dir / "pred_ANN.png")

    table = pd.DataFrame(rows); table.to_csv(out / "metrics.csv", index=False)
    return table
