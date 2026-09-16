from __future__ import annotations

import argparse
from pathlib import Path
from common import load_dataset


def arguments():
    p = argparse.ArgumentParser(description="DLC wear prediction study")
    p.add_argument("--data", required=True, help="XLSX/CSV dataset")
    p.add_argument("--sheet", default="DLC_database")
    p.add_argument("--study", nargs="+", choices=["data", "physics", "pgnn", "all"], default=["all"])
    p.add_argument("--formulation", nargs="+", choices=["dV", "k", "alpha", "all"], default=["all"])
    p.add_argument("--output", default="results")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-ann", action="store_true", help="Skip TensorFlow ANN (diagnostic only)")
    p.add_argument("--no-cv", action="store_true", help="Skip 5-fold CV")
    return p.parse_args()


def main():
    a = arguments(); df = load_dataset(a.data, a.sheet)
    studies = ["data", "physics", "pgnn"] if "all" in a.study else a.study
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)

    if "data" in studies:
        from data_driven import run
        print("\n=== Data-driven ===")
        print(run(df, out, a.seed, a.epochs, not a.no_ann, not a.no_cv).to_string(index=False))

    if "physics" in studies:
        from physics_constrained import run
        forms = ("dV", "k", "alpha") if "all" in a.formulation else tuple(a.formulation)
        print("\n=== Physics-constrained ===")
        print(run(df, out, forms, a.seed, a.epochs, not a.no_ann, not a.no_cv).to_string(index=False))

    if "pgnn" in studies:
        if a.no_ann:
            print("\nPGNN skipped: TensorFlow is required.")
        else:
            from pgnn import run
            print("\n=== PGNN ===")
            print(run(df, out, a.seed, a.epochs).to_string(index=False))


if __name__ == "__main__":
    main()
