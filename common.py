from __future__ import annotations

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

SCALE_FACTOR = 1e12
WEAR_MIN = 1e-12
WEAR_MAX = 1e-4

BASE_FEATURES = [
    "E_reduit (GPa)", "P_hertz (MPa)", "Sp2/Sp3", "Film hardness (GPa)",
    "Rq (nm)", "Ball hardness (GPa)", "Sliding velocity (m/s)",
    "Temperature", "Humidity", "Sliding distance (m)",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(path: str | Path, sheet_name: str = "DLC_database") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Supported dataset formats: .xlsx, .xls, .csv")


def drop_rows_with_missing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in columns if c in out.columns]
    if len(cols) != len(columns):
        missing = sorted(set(columns) - set(cols))
        raise KeyError(f"Missing required columns: {missing}")
    out[cols] = out[cols].replace([r"^\s*$", r"(?i)^nan$", r"(?i)^n/a$"], np.nan, regex=True)
    return out.dropna(subset=cols).copy()


def impute_by_group_mean(df: pd.DataFrame, target: str, groups: list[str]) -> pd.DataFrame:
    """Same group-mean logic as the research scripts."""
    out = df.copy()
    missing = [c for c in [target, *groups] if c not in out.columns]
    if missing:
        raise KeyError(f"Missing columns for group imputation: {missing}")
    out[target] = out[target].replace("", pd.NA)
    out[target] = pd.to_numeric(out[target], errors="coerce")
    group_means = out.groupby(groups, dropna=False)[target].mean()
    for key, mean in group_means.items():
        vals = key if isinstance(key, tuple) else (key,)
        group_mask = pd.Series(True, index=out.index)
        for col, val in zip(groups, vals):
            if pd.isna(val):
                group_mask &= out[col].isna()
            else:
                group_mask &= out[col].eq(val)
        if pd.isna(mean):
            out = out.loc[~group_mask].copy()
        else:
            out.loc[group_mask & out[target].isna(), target] = mean
    return out


def _numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")


def add_contact_quantities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = ["Film elastic modulus (GPa)", "Ball elastic modulus (GPa)", "Film Poisson",
            "Ball Poisson", "Load (N)", "Radius (mm)"]
    _numeric(out, cols)
    E1 = out["Film elastic modulus (GPa)"].to_numpy(float)
    E2 = out["Ball elastic modulus (GPa)"].to_numpy(float)
    nu1 = out["Film Poisson"].to_numpy(float)
    nu2 = out["Ball Poisson"].to_numpy(float)
    F = out["Load (N)"].to_numpy(float)
    R = out["Radius (mm)"].to_numpy(float)
    Ered = 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)
    a = ((3.0 * F * R) / (4.0 * Ered)) ** (1.0 / 3.0)
    out["E_reduit (GPa)"] = Ered
    out["Contact radius a (mm)"] = a
    out["A_real (mm2)"] = np.pi * a**2
    out["P_hertz (MPa)"] = (3.0 * F) / (2.0 * np.pi * a**2)
    return out


def prepare_comparison_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing order used by Phys_comp_equation.py."""
    out = impute_by_group_mean(df, "Rq (nm)", ["DLC type", "Process type"])
    out = impute_by_group_mean(out, "Film elastic modulus (GPa)", ["DLC type", "Process type"])
    out = impute_by_group_mean(out, "Film hardness (GPa)", ["DLC type"])
    out = drop_rows_with_missing(out, ["Wear rate", "Friction coefficient"])
    out = add_contact_quantities(out)
    _numeric(out, ["Wear rate", "Film hardness (GPa)", "Load (N)", "Friction coefficient"])
    K = out["Wear rate"].to_numpy(float)
    H = out["Film hardness (GPa)"].to_numpy(float)
    F = out["Load (N)"].to_numpy(float)
    mu = out["Friction coefficient"].to_numpy(float)
    out["dV"] = F * K
    out["k"] = H * K
    out["alpha"] = H * K / mu
    out = out[(out["Wear rate"] >= WEAR_MIN) & (out["Wear rate"] <= WEAR_MAX)]
    return out.reset_index(drop=True)


def prepare_pgnn_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing order used by Phys_PINN_v02.py."""
    out = impute_by_group_mean(df, "Rq (nm)", ["DLC type", "Process type"])
    out = drop_rows_with_missing(out, ["Wear rate", "Film hardness (GPa)", "Sliding distance (m)"])
    out = add_contact_quantities(out)
    _numeric(out, ["Wear rate", "Load (N)", "Sliding distance (m)", "Film hardness (GPa)"])
    K = out["Wear rate"].to_numpy(float)
    F = out["Load (N)"].to_numpy(float)
    L = out["Sliding distance (m)"].to_numpy(float)
    out["Wear volume (mm^3)"] = K * F * L
    out = out[(out["Wear rate"] >= WEAR_MIN) & (out["Wear rate"] <= WEAR_MAX)]
    return out.reset_index(drop=True)


def prepare_features(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    X = df[features].copy().replace(r"^\s*$", np.nan, regex=True)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    return np.asarray(X_scaled, dtype=np.float32), imputer, scaler


def target_to_log(y_phys: np.ndarray) -> np.ndarray:
    y = np.asarray(y_phys, dtype=float)
    return np.log10(y * SCALE_FACTOR + 1e-6)


def log_to_physical(y_log: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(y_log, dtype=float)) / SCALE_FACTOR


def split_indices(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    train, temp = train_test_split(idx, test_size=0.30, random_state=seed)
    test, val = train_test_split(temp, test_size=1.0 / 3.0, random_state=seed)
    return train, val, test


def mare(y_true, y_pred, eps: float = 1e-12) -> float:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(yt - yp) / (np.abs(yt) + eps)))


def metrics(y_true, y_pred, y_true_log=None, y_pred_log=None) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    out = {
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "R2": float(r2_score(yt, yp)),
        "MARE": mare(yt, yp),
    }
    if y_true_log is not None and y_pred_log is not None:
        yl = np.asarray(y_true_log, dtype=float).ravel()
        pl = np.asarray(y_pred_log, dtype=float).ravel()
        out.update({"RMSE_log": float(np.sqrt(mean_squared_error(yl, pl))),
                    "MAE_log": float(mean_absolute_error(yl, pl)),
                    "R2_log": float(r2_score(yl, pl))})
    return out


def save_predictions_plot(y_true, y_pred, title: str, path: Path) -> None:
    yt = np.asarray(y_true).ravel(); yp = np.asarray(y_pred).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp) & (yt > 0) & (yp > 0)
    if not mask.any():
        return
    lo = min(yt[mask].min(), yp[mask].min()); hi = max(yt[mask].max(), yp[mask].max())
    plt.figure(figsize=(5, 4)); plt.scatter(yt[mask], yp[mask], alpha=0.65)
    plt.plot([lo, hi], [lo, hi], "--"); plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Actual wear rate (mm³/Nm)"); plt.ylabel("Predicted wear rate (mm³/Nm)")
    plt.title(title); plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200); plt.close()


def sklearn_cv(models: dict, X: np.ndarray, y_log: np.ndarray, folds: int = 5) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y_log, cv=folds, scoring="neg_mean_squared_error")
        rmses = np.sqrt(-scores)
        rows.append({"model": name, "RMSE_log_cv": rmses.mean(), "RMSE_log_cv_std": rmses.std()})
    return pd.DataFrame(rows)
