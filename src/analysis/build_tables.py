"""
Build the analysis tables from a directory of run JSONs.

    python -m analysis.build_tables --runs-dir runs --out-dir tables

Output (all CSV, all tidy):

    per_input.csv           run x eps x input -- the join table
    eval_metrics.csv        run x eps, full eval split (population level)
    seed_metrics.csv        run x eps, certification subset + certification
    seed_score_metrics.csv  run x eps x score x variant x target (long)
    risk_coverage.csv       run x eps x score x variant x coverage
    curve_auc.csv           run, epsilon-integrated
    posterior_spread.csv    run x layer
    summary_*.csv           across-seed mean / SD / SE / 95% CI / n_valid
    paired_effects.csv      matched robust-vs-standard, per seed
    summary_effects.csv     across-seed summary of those effects
    field_coverage.csv      which plan-required eval fields were present

Nothing here plots. Figures read these tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis import seed_stats as S
from analysis.aggregate import paired_effects, summarize, summarize_effects
from analysis.loading import RUN_KEYS, field_coverage, load_all
from analysis.scores import DEFAULT_SCORES, SCORE_NAMES

_MERGE_KEYS = [*RUN_KEYS, "eps"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="tables")
    ap.add_argument(
        "--scores", nargs="+", default=list(DEFAULT_SCORES), choices=SCORE_NAMES,
        help="Scores entering the AUROC/AUPRC/Spearman loops. All five are "
             "always written to per_input.csv for sensitivity analysis.",
    )
    ap.add_argument("--taus", nargs="+", type=float, default=list(S.DEFAULT_TAUS))
    ap.add_argument("--coverages", nargs="+", type=float,
                    default=list(S.DEFAULT_COVERAGES))
    return ap.parse_args()


def _write(frame: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"  {label:<24} {len(frame):>8,} rows  ->  {path.name}")


def main():
    args = parse_args()
    out = Path(args.out_dir).expanduser().resolve()

    print(f"Loading runs from {args.runs_dir}")
    frames = load_all(Path(args.runs_dir))
    per_input = frames["per_input"]
    eval_metrics = frames["eval_metrics"]

    n_runs = per_input.groupby(RUN_KEYS, dropna=False).ngroups
    print(f"  {n_runs} runs, {per_input['seed'].nunique()} seeds, "
          f"families: {sorted(per_input['family'].unique())}")

    print("Within-seed statistics")
    subset = S.subset_outcomes(per_input)
    uncertainty = S.mean_uncertainty(per_input, scores=args.scores)
    certification = S.certification_summary(per_input, taus=args.taus)

    # One row per (run, eps). Certification is defined on a coarser epsilon
    # grid than PGD, so it joins as a left merge and leaves NaNs outside it.
    seed_metrics = subset.merge(uncertainty, on=_MERGE_KEYS, how="outer",
                                suffixes=("", "_dup"))
    seed_metrics = seed_metrics.merge(certification, on=_MERGE_KEYS,
                                      how="left", suffixes=("", "_cert"))
    seed_metrics = seed_metrics.loc[:, ~seed_metrics.columns.str.endswith("_dup")]

    score_metrics = S.detection_and_alignment(
        per_input, scores=args.scores, taus=args.taus
    )
    coverage = S.risk_coverage_table(
        per_input, scores=args.scores, coverages=args.coverages
    )
    auc = S.curve_auc(eval_metrics, subset)

    print("Across-seed summaries")
    summaries = {
        "summary_eval": summarize(eval_metrics),
        "summary_seed_metrics": summarize(seed_metrics),
        # Score tables carry their own facet keys, so those join the grouping.
        "summary_score_metrics": summarize(
            score_metrics, extra_keys=("eps", "score", "variant", "target")
        ),
        "summary_risk_coverage": summarize(
            coverage, extra_keys=("eps", "score", "variant", "coverage")
        ),
        "summary_curve_auc": summarize(auc, extra_keys=()),
    }

    print("Matched training effects")
    effect_frames = [
        paired_effects(eval_metrics),
        paired_effects(seed_metrics),
        paired_effects(score_metrics,
                       extra_keys=("eps", "score", "variant", "target")),
        paired_effects(auc, extra_keys=()),
    ]
    effects = pd.concat([f for f in effect_frames if not f.empty],
                        ignore_index=True)

    print(f"Writing tables to {out}")
    _write(per_input, out / "per_input.csv", "per_input")
    _write(eval_metrics, out / "eval_metrics.csv", "eval_metrics")
    _write(seed_metrics, out / "seed_metrics.csv", "seed_metrics")
    _write(score_metrics, out / "seed_score_metrics.csv", "seed_score_metrics")
    _write(coverage, out / "risk_coverage.csv", "risk_coverage")
    _write(auc, out / "curve_auc.csv", "curve_auc")
    _write(frames["posterior_spread"], out / "posterior_spread.csv",
           "posterior_spread")
    _write(frames["cert_aggregates"], out / "cert_aggregates_raw.csv",
           "cert_aggregates_raw")
    for name, frame in summaries.items():
        _write(frame, out / f"{name}.csv", name)
    _write(effects, out / "paired_effects.csv", "paired_effects")
    _write(summarize_effects(effects,
                             extra_keys=("eps", "score", "variant", "target")),
           out / "summary_effects.csv", "summary_effects")

    coverage_report = field_coverage(eval_metrics)
    _write(coverage_report, out / "field_coverage.csv", "field_coverage")
    missing = coverage_report.loc[~coverage_report["present"], "field"].tolist()
    if missing:
        print("\n  WARNING: clean_report did not emit these plan-required "
              f"fields: {missing}")
        print("  Calibration analyses (plan section B) will be incomplete.")


if __name__ == "__main__":
    main()
