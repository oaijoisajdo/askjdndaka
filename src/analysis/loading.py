"""
Turn the run JSONs into tidy frames.

One JSON holds BOTH conditions for one (family, seed, robust point), so the
standard block is repeated once per robust point. That duplication is real
and must be removed before any across-seed summary of standard-only curves,
or a family with three robust points will report each standard seed three
times and shrink its seed SE by sqrt(3). See ``dedupe_standard``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.scores import SCORE_NAMES, extract_scores, delta_scores

CONDITIONS = ("standard", "robust")

# Identify a unique trained model. train_eps / rob_lam are NaN for standard.
RUN_KEYS = ["family", "condition", "seed", "train_eps", "rob_lam"]

# Certification per-input fields carried through verbatim.
CERT_FIELDS = (
    "p_safe_lower", "p_safe_upper", "p_safe_point", "p_safe_point_optimistic",
    "p_safe_ci_low", "p_safe_ci_high",
    # Optimistic-side CP interval: without these the optimistic estimator's
    # interval is not reconstructable from the CSV (the two sides stop at
    # different n), which is what the bracketed analysis needs.
    "p_safe_ci_low_optimistic", "p_safe_ci_high_optimistic",
    "unknown_frac",
    # Raw per-side counts, so every reported bound can be recomputed from
    # the CSV alone rather than trusted.
    "k_violation_pessimistic", "unk_pessimistic",
    "k_violation_optimistic", "unk_optimistic",
    "n_samples_used", "n_property_evaluations", "mean_net_verdict",
)


def iter_run_files(root: Path):
    return sorted(Path(root).rglob("*.json"))


def _scalars(report: dict) -> dict:
    """Numeric top-level fields only, so nested per-input blocks are skipped."""
    return {
        k: float(v) for k, v in report.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _keys(payload: dict, condition: str) -> dict:
    exp = payload["experiment"]
    point = payload.get("configuration", {}).get("robust_point", {})
    robust = condition == "robust"
    return {
        "family": exp["family"],
        "condition": condition,
        "seed": int(exp["seed"]),
        "train_eps": float(point.get("epsilon", np.nan)) if robust else np.nan,
        "rob_lam": float(point.get("rob_lam", np.nan)) if robust else np.nan,
        "eval_split": exp.get("eval_split"),
        "run_file": exp.get("_source", ""),
    }


def _with_expected_entropy(row: dict) -> dict:
    """Fill ``mean_expected_entropy`` when ``clean_report`` omitted it.

    Expected (aleatoric) entropy is E_S[H(p_S)] = H(p_bar) - MI by
    definition, and the mean over inputs is linear, so the population-level
    value is exactly ``mean_predictive_entropy - mean_mutual_information``.
    This is an identity, not an approximation: it agrees with the
    directly-computed per-input quantity to float accumulation noise (~7e-9).

    Runs produced before the emission was added to ``clean_report`` are
    therefore fully recoverable and need not be re-run. Rows that already
    carry the field are left untouched.
    """
    if ("mean_expected_entropy" not in row
            and "mean_predictive_entropy" in row
            and "mean_mutual_information" in row):
        row["mean_expected_entropy"] = (row["mean_predictive_entropy"]
                                        - row["mean_mutual_information"])
    return row


def eval_rows(payload: dict) -> list[dict]:
    """
    Population-level metrics on the full eval split, one row per epsilon.

    Whatever ``clean_report`` emits is carried through unchanged; the required
    set (accuracy, NLL, Brier, ECE, mean entropy/MI/confidence) is checked
    once in ``field_coverage`` rather than assumed here. eps == 0 is the clean
    block, which is what puts the clean point on the accuracy-vs-eps curve.
    """
    rows = []
    for condition in CONDITIONS:
        block = payload.get(condition)
        if block is None:
            continue
        keys = _keys(payload, condition)
        rows.append(_with_expected_entropy(
            {**keys, "eps": 0.0, **_scalars(block["clean"])}))
        for eps_key, report in block.get("pgd", {}).items():
            rows.append(_with_expected_entropy(
                {**keys, "eps": float(eps_key), **_scalars(report)}))
    return rows


def per_input_rows(payload: dict) -> list[dict]:
    """
    The join table: one row per (run, epsilon, certification-subset input).

    Clean uncertainty is constant in epsilon and is repeated on every row so
    that clean, attacked and delta scores sit side by side without a second
    join. Certification columns are NaN at epsilons outside the certification
    grid, since PGD covers a wider grid than the verifier.
    """
    rows = []
    for condition in CONDITIONS:
        block = payload.get(condition)
        if block is None:
            continue
        keys = _keys(payload, condition)
        clean_block = block["cert_subset_uncertainty"]
        n = len(clean_block["correct"])
        eval_idx = payload["experiment"].get("certification_indices") or list(range(n))

        clean = extract_scores(clean_block, n)
        clean_correct = np.asarray(clean_block["correct"], dtype=float)
        certification = block.get("certification", {})

        for eps_key, report in block.get("pgd", {}).items():
            adv_block = report.get("cert_subset_uncertainty")
            adv = extract_scores(adv_block, n)
            delta = delta_scores(clean, adv)
            adv_correct = (
                np.asarray(adv_block["correct"], dtype=float)
                if adv_block else np.full(n, np.nan)
            )
            cert = certification.get(eps_key, {}).get("per_input", {})

            for i in range(n):
                row = {
                    **keys,
                    "eps": float(eps_key),
                    "input_idx": i,
                    "eval_index": int(eval_idx[i]),
                    # Only the certification block records the label; NaN at
                    # epsilons the verifier did not run on.
                    "true_label": int(cert["true_label"][i]) if cert else np.nan,
                    "correct_clean": clean_correct[i],
                    "correct_adv": adv_correct[i],
                    # Adversarial error: the event analysis C detects.
                    "adv_error": 1.0 - adv_correct[i],
                    # Attack success: conditional on the clean prediction
                    # having been right. Undefined where it was already wrong.
                    "attack_success": (
                        1.0 - adv_correct[i] if clean_correct[i] == 1 else np.nan
                    ),
                }
                for name in SCORE_NAMES:
                    row[f"{name}_clean"] = clean[name][i]
                    row[f"{name}_adv"] = adv[name][i]
                    row[f"{name}_delta"] = delta[name][i]
                for field in CERT_FIELDS:
                    values = cert.get(field)
                    row[field] = values[i] if values is not None else np.nan
                rows.append(row)
    return rows


def cert_aggregate_rows(payload: dict) -> list[dict]:
    """Verifier aggregates already computed in the run, kept as-is."""
    rows = []
    for condition in CONDITIONS:
        block = payload.get(condition)
        if block is None:
            continue
        keys = _keys(payload, condition)
        for eps_key, agg in block.get("certification", {}).items():
            rows.append({
                **keys, "eps": float(eps_key),
                **{k: v for k, v in agg.items() if k != "per_input"},
            })
    return rows


def sigma_rows(payload: dict) -> list[dict]:
    """Layerwise posterior spread, for the analysis-F mechanism check."""
    rows = []
    for condition in CONDITIONS:
        stats = payload.get("posterior_sigma", {}).get(condition)
        if not stats or not stats.get("layers"):
            continue
        keys = _keys(payload, condition)
        for layer in stats["layers"]:
            rows.append({**keys, "posterior_kind": stats.get("kind"),
                         "p_drop": stats.get("p_drop", np.nan), **layer})
    return rows


def load_all(root: Path) -> dict[str, pd.DataFrame]:
    per_input, evaluation, cert_agg, sigma = [], [], [], []
    for path in iter_run_files(root):
        payload = json.loads(path.read_text())
        if "experiment" not in payload:
            continue                                   # not a run file
        payload["experiment"]["_source"] = str(path.name)
        per_input += per_input_rows(payload)
        evaluation += eval_rows(payload)
        cert_agg += cert_aggregate_rows(payload)
        sigma += sigma_rows(payload)

    if not evaluation:
        raise FileNotFoundError(f"No run JSONs with an 'experiment' block under {root}")

    return {
        "per_input": pd.DataFrame(per_input),
        "eval_metrics": pd.DataFrame(evaluation),
        "cert_aggregates": pd.DataFrame(cert_agg),
        "posterior_spread": pd.DataFrame(sigma),
    }


def dedupe_standard(df: pd.DataFrame, extra_keys=("eps",)) -> pd.DataFrame:
    """
    Collapse the standard block's repetition across robust points.

    The standard model for a given (family, seed) is identical in every file,
    so keeping one copy is lossless. Robust rows are untouched -- they differ
    by (train_eps, rob_lam) and are genuine distinct runs.
    """
    keys = ["family", "seed", *extra_keys]
    standard = df[df["condition"] == "standard"].drop_duplicates(subset=keys)
    robust = df[df["condition"] == "robust"]
    return pd.concat([standard, robust], ignore_index=True)


REQUIRED_EVAL_FIELDS = (
    "accuracy", "nll", "brier", "ece",
    "mean_predictive_entropy", "mean_expected_entropy",
    "mean_mutual_information", "mean_confidence",
)


def field_coverage(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Which plan-required eval fields ``clean_report`` actually produced.

    Reported rather than enforced: the analysis still builds without NLL or
    Brier, it just cannot populate the calibration figures, and it is better
    to see that in a coverage table than to discover an empty panel later.
    """
    return pd.DataFrame([
        {"field": f, "present": f in eval_df.columns,
         "non_null": int(eval_df[f].notna().sum()) if f in eval_df.columns else 0}
        for f in REQUIRED_EVAL_FIELDS
    ])
