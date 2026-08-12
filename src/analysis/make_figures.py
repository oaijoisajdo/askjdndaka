#!/usr/bin/env python3
"""
Render every thesis figure from the build_tables output.

Usage, from the repository root:

    python3 -m src.analysis.make_figures                       # tables/ -> figures/
    python3 -m src.analysis.make_figures --tables tables --out figures
    python3 -m src.analysis.make_figures --format pdf          # vector, for LaTeX
    python3 -m src.analysis.make_figures --only A1 A3          # a subset
    python3 -m src.analysis.make_figures --tau 0.5             # sensitivity pass

The primary-endpoint figure (A1) conditions on the clean predictive margin
when that column is populated, and falls back to confidence otherwise --
in which case the figure is titled and filed as PILOT, because a
confidence-conditioned partial correlation is not the pre-registered endpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from analysis import alignment as al
from analysis import figures as F


def build(tables: Path, out: Path, fmt: str, tau: float, only: list[str] | None):
    out.mkdir(parents=True, exist_ok=True)
    cm, pm = F.load_model_metrics(str(tables))
    cert = al.load_per_input(tables / "per_input.csv")

    margin_ok = cert["one_minus_margin_clean"].notna().any()
    baseline = ("one_minus_margin_clean" if margin_ok
                else "one_minus_confidence_clean")
    pilot = not margin_ok
    if pilot:
        print("  ! predictive_margin is empty -> A1 uses the confidence "
              "baseline and is written as PILOT (not the frozen endpoint).")

    a1_name = "A1_partial_rho" + ("_PILOT" if pilot else "")
    jobs = {
        "D1": ("D1_accuracy", lambda: F.fig_d1_accuracy(cm, pm)),
        "D2": ("D2_uncertainty_vs_eps", lambda: F.fig_d2_uncertainty(cm, pm)),
        "D3": ("D3_calibration_vs_eps", lambda: F.fig_d3_calibration(cm, pm)),
        "D4": ("D4_certified_share", lambda: F.fig_d4_certified(cert, tau)),
        "D4b": ("D4b_bound_comparison",
                lambda: F.fig_d4b_bound_comparison(cert, tau)),
        "A1": (a1_name, lambda: F.fig_a1_partial_rho(cert, baseline, pilot)),
        "A1b": (a1_name.replace("A1", "A1b") + "_forest",
                lambda: F.fig_a1b_forest(cert, baseline)),
        "A2": ("A2_bracket_auroc", lambda: F.fig_a2_bracket_auroc(cert, tau=tau)),
        "A3": ("A3_conservatism", lambda: F.fig_a3_conservatism(cert, tau)),
        "A4": ("A4_censoring", lambda: F.fig_a4_censoring(cert)),
        "A5": ("A5_composition", lambda: F.fig_a5_composition(cert, tau)),
        "A6": ("A6_scatter", lambda: F.fig_a6_scatter(cert, tau=tau)),
    }
    if only:
        missing = [k for k in only if k not in jobs]
        if missing:
            raise SystemExit(f"unknown figure id(s): {missing}. "
                             f"choose from {sorted(jobs)}")
        jobs = {k: v for k, v in jobs.items() if k in only}

    suffix = f"_tau{tau:g}".replace(".", "p") if tau != 0.9 else ""
    for fid, (name, fn) in jobs.items():
        try:
            fig = fn()
        except Exception as exc:                       # keep going
            print(f"  {fid:3s} FAILED: {type(exc).__name__}: {exc}")
            continue
        path = out / f"{name}{suffix}.{fmt}"
        fig.savefig(path)
        print(f"  {fid:3s} -> {path}")
        import matplotlib.pyplot as plt
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="tables", type=Path)
    ap.add_argument("--out", default="figures", type=Path)
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--tau", default=0.9, type=float)
    ap.add_argument("--only", nargs="*", default=None,
                    help="figure ids, e.g. --only A1 A3")
    a = ap.parse_args()
    print(f"Reading tables from {a.tables.resolve()}")
    build(a.tables, a.out, a.format, a.tau, a.only)


if __name__ == "__main__":
    main()