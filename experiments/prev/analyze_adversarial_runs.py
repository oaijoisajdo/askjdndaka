#!/usr/bin/env python3
"""Analyze JSON files written by ``experiments.adversarial_run``.

The script keeps three notions separate:

* empirical robustness of the posterior-predictive classifier (PGD),
* posterior mass on individually safe networks (P_safe), and
* verifier incompleteness (unknown outcomes).

It creates tidy CSV tables, seed-level alignment metrics, paired
standard-versus-robust effects, quality-control diagnostics, figures, and a
short Markdown report.  Metrics are always computed within a seed.  The script
does not pool input rows across seeds when estimating correlations or AUROC.

Example
-------
python experiments/analyze_adversarial_runs.py \
    --runs-dir runs \
    --out-dir analysis/adversarial \
    --tau 0.5 --tau 0.9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

# Avoid failures on headless/read-only systems before importing pyplot.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bnn-analysis-mpl")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


BASE_META = [
    "architecture",
    "family",
    "seed",
    "eval_split",
    "training",
    "train_epsilon",
    "rob_lam",
    "config_id",
    "config_label",
    "model_id",
    "cert_theta",
    "cert_gamma",
    "cert_alpha",
]

RISK_SCORE_SOURCES = {
    "predictive_entropy": ("predictive_entropy", False),
    "mutual_information": ("mutual_information", False),
    "expected_entropy": ("expected_entropy", False),
    "one_minus_confidence": ("confidence", True),
    "one_minus_margin": ("predictive_margin", True),
}


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def scalar_metrics(block: Mapping[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in block.items() if finite_number(v)}


def number_tag(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "na"
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean_sd_ci(values: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return {"n_seeds": 0, "mean": math.nan, "sd": math.nan,
                "ci_low": math.nan, "ci_high": math.nan}
    mean = float(x.mean())
    if len(x) == 1:
        return {"n_seeds": 1, "mean": mean, "sd": math.nan,
                "ci_low": math.nan, "ci_high": math.nan}
    sd = float(x.std(ddof=1))
    half = float(stats.t.ppf((1 + confidence) / 2, len(x) - 1) * sd / np.sqrt(len(x)))
    return {"n_seeds": int(len(x)), "mean": mean, "sd": sd,
            "ci_low": mean - half, "ci_high": mean + half}


def normalized_auc(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[keep], y_arr[keep]
    if len(x_arr) < 2:
        return math.nan
    order = np.argsort(x_arr)
    x_arr, y_arr = x_arr[order], y_arr[order]
    width = x_arr[-1] - x_arr[0]
    if width <= 0:
        return math.nan
    return float(np.trapezoid(y_arr, x_arr) / width)


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan, math.nan, int(len(x))
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue), int(len(x))


def binary_metrics(score: np.ndarray, target: np.ndarray) -> dict[str, float]:
    keep = np.isfinite(score) & np.isfinite(target)
    score, target = score[keep], target[keep].astype(int)
    prevalence = float(target.mean()) if len(target) else math.nan
    if len(target) < 2 or np.unique(target).size < 2:
        return {"n": int(len(target)), "prevalence": prevalence,
                "auroc": math.nan, "auprc": math.nan}
    return {
        "n": int(len(target)),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(target, score)),
        "auprc": float(average_precision_score(target, score)),
    }


def verdict_class(lower: float, upper: float, tau: float) -> str:
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return "missing"
    if lower >= tau:
        return "robust_above_threshold"
    if upper < tau:
        return "below_threshold"
    return "inconclusive"


@dataclass
class AnalysisData:
    inventory: list[dict[str, Any]] = field(default_factory=list)
    clean: list[dict[str, Any]] = field(default_factory=list)
    pgd: list[dict[str, Any]] = field(default_factory=list)
    certification: list[dict[str, Any]] = field(default_factory=list)
    per_input: list[dict[str, Any]] = field(default_factory=list)
    posterior_spread: list[dict[str, Any]] = field(default_factory=list)
    paired_effects: list[dict[str, Any]] = field(default_factory=list)
    spread_effects: list[dict[str, Any]] = field(default_factory=list)
    pairs: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)

    def check(self, level: str, path: str, check: str, detail: str) -> None:
        self.checks.append({"level": level, "file": path, "check": check,
                            "detail": detail})


def model_meta(payload: Mapping[str, Any], training: str) -> dict[str, Any]:
    exp = payload["experiment"]
    configuration = payload.get("configuration", {})
    point = configuration.get("robust_point", {})
    cert_config = configuration.get("cert", {})
    eps = safe_float(point.get("epsilon")) if training == "robust" else math.nan
    lam = safe_float(point.get("rob_lam")) if training == "robust" else math.nan
    config_id = (
        "standard" if training == "standard"
        else f"robust_eps{number_tag(eps)}_lam{number_tag(lam)}"
    )
    config_label = (
        "standard" if training == "standard"
        else rf"robust ($\epsilon_{{train}}={eps:g}$, $\lambda={lam:g}$)"
    )
    architecture = str(exp.get("architecture", "unknown"))
    family = str(exp.get("family", "unknown"))
    seed = int(exp.get("seed", -1))
    split = str(exp.get("eval_split", "unknown"))
    model_id = f"{architecture}|{family}|s{seed}|{split}|{config_id}"
    return {
        "architecture": architecture,
        "family": family,
        "seed": seed,
        "eval_split": split,
        "training": training,
        "train_epsilon": eps,
        "rob_lam": lam,
        "config_id": config_id,
        "config_label": config_label,
        "model_id": model_id,
        "cert_theta": safe_float(cert_config.get("theta")),
        "cert_gamma": safe_float(cert_config.get("gamma")),
        "cert_alpha": safe_float(cert_config.get("alpha")),
    }


def validate_payload(payload: Mapping[str, Any], path: Path) -> None:
    missing = {"experiment", "configuration", "standard", "robust"} - set(payload)
    if missing:
        raise ValueError(f"{path}: missing top-level keys {sorted(missing)}")
    for side in ("standard", "robust"):
        if "clean" not in payload[side] or "cert_subset_uncertainty" not in payload[side]:
            raise ValueError(f"{path}: {side} lacks clean or per-input uncertainty data")


def side_fingerprint(side: Mapping[str, Any], sigma: Mapping[str, Any]) -> str:
    return canonical_hash({"report": side, "posterior_sigma": sigma})


def add_posterior_spread(
    data: AnalysisData,
    meta: Mapping[str, Any],
    spread: Mapping[str, Any] | None,
) -> None:
    if not spread:
        return
    kind = spread.get("kind", "unknown")
    for layer in spread.get("layers") or []:
        row = {**meta, "posterior_kind": kind}
        row.update({k: v for k, v in spread.items() if finite_number(v) and k != "layers"})
        row.update({k: v for k, v in layer.items() if np.isscalar(v)})
        data.posterior_spread.append(row)


def uncertainty_length(unc: Mapping[str, Any]) -> int:
    for key in ("correct", "predictive_entropy", "confidence"):
        value = unc.get(key)
        if isinstance(value, list):
            return len(value)
    raise ValueError("No per-input array found in cert_subset_uncertainty")


def array_from(block: Mapping[str, Any] | None, key: str, n: int) -> np.ndarray:
    if not block or key not in block:
        return np.full(n, np.nan)
    value = np.asarray(block[key])
    if value.ndim != 1 or len(value) != n:
        raise ValueError(f"{key} has shape {value.shape}; expected ({n},)")
    try:
        return value.astype(float)
    except (TypeError, ValueError):
        return value.astype(object)


def add_side(
    data: AnalysisData,
    payload: Mapping[str, Any],
    side_name: str,
    path: Path,
    taus: Sequence[float],
) -> None:
    side = payload[side_name]
    meta = model_meta(payload, side_name)
    data.clean.append({**meta, **scalar_metrics(side.get("clean", {}))})

    for eps_key, report in side.get("pgd", {}).items():
        data.pgd.append({**meta, "eval_epsilon": float(eps_key),
                         **scalar_metrics(report)})

    for eps_key, report in side.get("certification", {}).items():
        summary = {k: v for k, v in scalar_metrics(report).items() if k != "n_inputs"}
        data.certification.append({**meta, "eval_epsilon": float(eps_key), **summary})

    add_posterior_spread(
        data, meta, payload.get("posterior_sigma", {}).get(side_name)
    )

    unc = side["cert_subset_uncertainty"]
    n = uncertainty_length(unc)
    eps_keys = set(side.get("certification", {})) | set(side.get("cert_subset_pgd_hit", {}))
    if not eps_keys:
        data.check("warning", str(path), "no_per_input_robustness",
                   f"{side_name}: uncertainty exists but neither certification nor subset PGD exists")
        return

    correct = array_from(unc, "correct", n)
    entropy = array_from(unc, "predictive_entropy", n)
    mi = array_from(unc, "mutual_information", n)
    expected_entropy = array_from(unc, "expected_entropy", n)
    confidence = array_from(unc, "confidence", n)
    margin_key = "predictive_margin" if "predictive_margin" in unc else "margin"
    margin = array_from(unc, margin_key, n)

    for eps_key in sorted(eps_keys, key=float):
        eps = float(eps_key)
        cert = side.get("certification", {}).get(eps_key, {}).get("per_input")
        hit_block = side.get("cert_subset_pgd_hit", {})
        hit = array_from(hit_block, eps_key, n)
        true_label = array_from(cert, "true_label", n)
        p_lower = array_from(cert, "p_safe_lower", n)
        p_upper = array_from(cert, "p_safe_upper", n)
        p_point = array_from(cert, "p_safe_point", n)
        p_opt = array_from(cert, "p_safe_point_optimistic", n)
        ci_low = array_from(cert, "p_safe_ci_low", n)
        ci_high = array_from(cert, "p_safe_ci_high", n)
        unknown = array_from(cert, "unknown_frac", n)
        n_samples = array_from(cert, "n_samples_used", n)
        n_evals = array_from(cert, "n_property_evaluations", n)
        verdict = array_from(cert, "mean_net_verdict", n)

        for i in range(n):
            row = {
                **meta,
                "eval_epsilon": eps,
                "input_index": i,
                "true_label": true_label[i],
                "clean_correct": correct[i],
                "predictive_entropy": entropy[i],
                "mutual_information": mi[i],
                "expected_entropy": expected_entropy[i],
                "confidence": confidence[i],
                "predictive_margin": margin[i],
                "pgd_hit": hit[i],
                "pgd_attack_success": hit[i] if correct[i] == 1 else math.nan,
                "p_safe_lower": p_lower[i],
                "p_safe_upper": p_upper[i],
                "p_safe_point": p_point[i],
                "p_safe_point_optimistic": p_opt[i],
                "p_safe_ci_low": ci_low[i],
                "p_safe_ci_high": ci_high[i],
                "unknown_frac": unknown[i],
                "n_samples_used": n_samples[i],
                "n_property_evaluations": n_evals[i],
                "mean_net_verdict": verdict[i],
            }
            for tau in taus:
                row[f"cert_outcome_tau_{number_tag(tau)}"] = verdict_class(
                    safe_float(p_lower[i]), safe_float(p_upper[i]), tau
                )
            data.per_input.append(row)

        if cert:
            finite = np.isfinite(p_point) & np.isfinite(p_opt) & np.isfinite(unknown)
            if finite.any():
                mae = float(np.mean(np.abs((p_opt[finite] - p_point[finite]) - unknown[finite])))
                if mae > 1e-6:
                    data.check("warning", str(path), "unknown_mass_identity",
                               f"{side_name}, eps={eps:g}: MAE={mae:.3g}")
            if np.any(np.isfinite(p_lower) & np.isfinite(p_upper) & (p_lower > p_upper + 1e-12)):
                data.check("error", str(path), "bound_order",
                           f"{side_name}, eps={eps:g}: lower bound exceeds upper bound")
            if np.any(np.isfinite(ci_low) & np.isfinite(ci_high) & (ci_low > ci_high + 1e-12)):
                data.check("error", str(path), "ci_order",
                           f"{side_name}, eps={eps:g}: CI low exceeds CI high")


def add_pair_effects(
    data: AnalysisData,
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    std_meta, rob_meta = model_meta(payload, "standard"), model_meta(payload, "robust")
    pair_meta = {
        "comparison_id": str(path),
        "architecture": rob_meta["architecture"],
        "family": rob_meta["family"],
        "seed": rob_meta["seed"],
        "eval_split": rob_meta["eval_split"],
        "train_epsilon": rob_meta["train_epsilon"],
        "rob_lam": rob_meta["rob_lam"],
        "robust_config_id": rob_meta["config_id"],
        "standard_model_id": std_meta["model_id"],
        "robust_model_id": rob_meta["model_id"],
    }
    data.pairs.append(pair_meta)
    standard, robust = payload["standard"], payload["robust"]

    def add_block(group: str, eps: float, a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
        for metric in sorted(set(scalar_metrics(a)) & set(scalar_metrics(b))):
            if metric in {"n_inputs", "n_mc_samples"}:
                continue
            av, bv = float(a[metric]), float(b[metric])
            data.paired_effects.append({
                **pair_meta, "outcome_group": group, "eval_epsilon": eps,
                "metric": metric, "standard": av, "robust": bv,
                "delta": bv - av,
            })

    add_block("clean", math.nan, standard.get("clean", {}), robust.get("clean", {}))
    for eps_key in sorted(set(standard.get("pgd", {})) & set(robust.get("pgd", {})), key=float):
        add_block("pgd", float(eps_key), standard["pgd"][eps_key], robust["pgd"][eps_key])
    for eps_key in sorted(
        set(standard.get("certification", {})) & set(robust.get("certification", {})),
        key=float,
    ):
        add_block(
            "certification", float(eps_key),
            standard["certification"][eps_key], robust["certification"][eps_key],
        )

    std_layers = payload.get("posterior_sigma", {}).get("standard", {}).get("layers") or []
    rob_layers = payload.get("posterior_sigma", {}).get("robust", {}).get("layers") or []
    for layer_a, layer_b in zip(std_layers, rob_layers):
        for metric in sorted(set(scalar_metrics(layer_a)) & set(scalar_metrics(layer_b))):
            if metric == "layer":
                continue
            av, bv = float(layer_a[metric]), float(layer_b[metric])
            data.spread_effects.append({
                **pair_meta, "layer": int(layer_a.get("layer", -1)), "metric": metric,
                "standard": av, "robust": bv, "delta": bv - av,
            })


def check_monotonicity(data: AnalysisData, pgd: pd.DataFrame, cert: pd.DataFrame) -> None:
    if not pgd.empty and "accuracy" in pgd:
        for model_id, group in pgd.groupby("model_id", dropna=False):
            group = group.sort_values("eval_epsilon")
            upward = np.diff(group["accuracy"].to_numpy(float))
            if len(upward) and np.nanmax(upward) > 0.02:
                data.check("warning", model_id, "pgd_nonmonotonic",
                           f"accuracy rises by up to {np.nanmax(upward):.3f} as epsilon increases")
    if not cert.empty and "mean_p_safe_point" in cert:
        for model_id, group in cert.groupby("model_id", dropna=False):
            group = group.sort_values("eval_epsilon")
            upward = np.diff(group["mean_p_safe_point"].to_numpy(float))
            if len(upward) and np.nanmax(upward) > 0.05:
                data.check("warning", model_id, "p_safe_nonmonotonic",
                           f"mean point estimate rises by up to {np.nanmax(upward):.3f}")


def load_runs(runs_dir: Path, taus: Sequence[float]) -> AnalysisData:
    paths = sorted(runs_dir.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON files found below {runs_dir}")

    data = AnalysisData()
    seen_models: dict[str, str] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate_payload(payload, path)
        except Exception as exc:
            data.check("error", str(path), "load_or_schema", str(exc))
            continue

        exp = payload["experiment"]
        robust_meta = model_meta(payload, "robust")
        pair_is_new = robust_meta["model_id"] not in seen_models
        data.inventory.append({
            "file": str(path),
            "architecture": exp.get("architecture"),
            "family": exp.get("family"),
            "seed": exp.get("seed"),
            "eval_split": exp.get("eval_split"),
            "eval_inputs": exp.get("eval_inputs"),
            "certification_inputs": exp.get("certification_inputs"),
            "mc_samples": exp.get("mc_samples"),
            "pgd_enabled": exp.get("pgd_enabled"),
            "certification_enabled": exp.get("certification_enabled"),
            "train_epsilon": robust_meta["train_epsilon"],
            "rob_lam": robust_meta["rob_lam"],
            "robust_config_id": robust_meta["config_id"],
            "cert_theta": robust_meta["cert_theta"],
            "cert_gamma": robust_meta["cert_gamma"],
            "cert_alpha": robust_meta["cert_alpha"],
        })

        # Standard reports are duplicated in every robust-sweep JSON.  Verify
        # equality before keeping only one descriptive copy.
        for side_name in ("standard", "robust"):
            meta = model_meta(payload, side_name)
            fingerprint = side_fingerprint(
                payload[side_name], payload.get("posterior_sigma", {}).get(side_name, {})
            )
            previous = seen_models.get(meta["model_id"])
            if previous is None:
                seen_models[meta["model_id"]] = fingerprint
                try:
                    add_side(data, payload, side_name, path, taus)
                except Exception as exc:
                    data.check("error", str(path), "parse_side", f"{side_name}: {exc}")
            elif previous != fingerprint:
                data.check(
                    "error", str(path), "duplicate_model_mismatch",
                    f"{meta['model_id']} differs across JSON files; first copy retained",
                )
        if pair_is_new:
            add_pair_effects(data, payload, path)
        else:
            data.check("warning", str(path), "duplicate_pair_skipped",
                       f"paired effects for {robust_meta['model_id']} were already loaded")

    if not data.inventory:
        errors = "; ".join(c["detail"] for c in data.checks[:3])
        raise ValueError(f"No valid experiment payloads were loaded. {errors}")
    return data


def risk_scores(group: pd.DataFrame) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for score_name, (column, invert) in RISK_SCORE_SOURCES.items():
        if column not in group or group[column].notna().sum() == 0:
            continue
        values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
        result[score_name] = 1.0 - values if invert else values
    return result


def alignment_tables(
    per_input: pd.DataFrame, taus: Sequence[float]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    correlations: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    group_cols = BASE_META + ["eval_epsilon"]

    for keys, full in per_input.groupby(group_cols, dropna=False, sort=False):
        base = dict(zip(group_cols, keys))
        for subset_name, group in (
            ("all", full),
            ("clean_correct", full[full["clean_correct"] == 1]),
        ):
            if group.empty:
                continue
            for score_name, score in risk_scores(group).items():
                p_pess = pd.to_numeric(group["p_safe_point"], errors="coerce").to_numpy(float)
                p_opt = pd.to_numeric(
                    group["p_safe_point_optimistic"], errors="coerce"
                ).to_numpy(float)
                rho_p, pval_p, n_p = spearman(score, p_pess)
                rho_o, pval_o, n_o = spearman(score, p_opt)
                correlations.append({
                    **base, "subset": subset_name, "score": score_name,
                    "n_pessimistic": n_p, "rho_pessimistic": rho_p,
                    "pvalue_pessimistic": pval_p,
                    "n_optimistic": n_o, "rho_optimistic": rho_o,
                    "pvalue_optimistic": pval_o,
                })

                pgd = pd.to_numeric(group["pgd_hit"], errors="coerce").to_numpy(float)
                bm = binary_metrics(score, pgd)
                detections.append({
                    **base, "subset": subset_name, "score": score_name,
                    "task": "pgd_failure", "tau": math.nan, **bm,
                })

                for tau in taus:
                    outcome_col = f"cert_outcome_tau_{number_tag(tau)}"
                    labels = group[outcome_col].map({
                        "robust_above_threshold": 0.0,
                        "below_threshold": 1.0,
                    }).to_numpy(float)
                    bm = binary_metrics(score, labels)
                    detections.append({
                        **base, "subset": subset_name, "score": score_name,
                        "task": "certified_below_threshold", "tau": tau, **bm,
                    })

        for tau in taus:
            column = f"cert_outcome_tau_{number_tag(tau)}"
            counts = full[column].value_counts(dropna=False)
            total = len(full)
            outcomes.append({
                **base, "tau": tau, "n": total,
                "n_robust_above": int(counts.get("robust_above_threshold", 0)),
                "n_below": int(counts.get("below_threshold", 0)),
                "n_inconclusive": int(counts.get("inconclusive", 0)),
                "n_missing": int(counts.get("missing", 0)),
                "frac_robust_above": float(counts.get("robust_above_threshold", 0) / total),
                "frac_below": float(counts.get("below_threshold", 0) / total),
                "frac_inconclusive": float(counts.get("inconclusive", 0) / total),
                "frac_missing": float(counts.get("missing", 0) / total),
            })

    return pd.DataFrame(correlations), pd.DataFrame(detections), pd.DataFrame(outcomes)


def certificate_diagnostics(per_input: pd.DataFrame) -> pd.DataFrame:
    """Aggregate verifier censoring and precision without reclassifying unknowns."""
    rows: list[dict[str, Any]] = []
    group_cols = BASE_META + ["eval_epsilon"]
    for keys, group in per_input.groupby(group_cols, dropna=False, sort=False):
        base = dict(zip(group_cols, keys))
        lower = pd.to_numeric(group["p_safe_lower"], errors="coerce").to_numpy(float)
        upper = pd.to_numeric(group["p_safe_upper"], errors="coerce").to_numpy(float)
        ci_low = pd.to_numeric(group["p_safe_ci_low"], errors="coerce").to_numpy(float)
        ci_high = pd.to_numeric(group["p_safe_ci_high"], errors="coerce").to_numpy(float)
        unknown = pd.to_numeric(group["unknown_frac"], errors="coerce").to_numpy(float)
        evaluations = pd.to_numeric(
            group["n_property_evaluations"], errors="coerce"
        ).to_numpy(float)
        finite_bounds = np.isfinite(lower) & np.isfinite(upper)
        theta = safe_float(base.get("cert_theta"))
        ceiling = 1.0 - theta if np.isfinite(theta) else 1.0
        rows.append({
            **base,
            "n_inputs": int(len(group)),
            "n_with_bounds": int(finite_bounds.sum()),
            "mean_bound_width": (
                float(np.mean(upper[finite_bounds] - lower[finite_bounds]))
                if finite_bounds.any() else math.nan
            ),
            "mean_cp_ci_width": (
                float(np.nanmean(ci_high - ci_low))
                if np.isfinite(ci_high - ci_low).any() else math.nan
            ),
            "mean_unknown_frac": (
                float(np.nanmean(unknown)) if np.isfinite(unknown).any() else math.nan
            ),
            "frac_lower_at_zero": (
                float(np.mean(np.isclose(lower[finite_bounds], 0.0, atol=1e-12)))
                if finite_bounds.any() else math.nan
            ),
            "upper_bound_ceiling": ceiling,
            "frac_upper_at_ceiling": (
                float(np.mean(np.isclose(upper[finite_bounds], ceiling, atol=1e-10)))
                if finite_bounds.any() else math.nan
            ),
            "mean_property_evaluations": (
                float(np.nanmean(evaluations))
                if np.isfinite(evaluations).any() else math.nan
            ),
        })
    return pd.DataFrame(rows)


def effect_summary(effects: pd.DataFrame) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame()
    group_cols = [
        "architecture", "family", "eval_split", "train_epsilon", "rob_lam",
        "robust_config_id", "outcome_group", "eval_epsilon", "metric",
    ]
    rows = []
    for keys, group in effects.groupby(group_cols, dropna=False):
        rows.append({**dict(zip(group_cols, keys)), **mean_sd_ci(group["delta"])})
    return pd.DataFrame(rows)


def seed_summary(df: pd.DataFrame, value_columns: Sequence[str], extra: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = [
        "architecture", "family", "eval_split", "training", "train_epsilon",
        "rob_lam", "config_id", "eval_epsilon", *extra,
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        for metric in value_columns:
            if metric not in group:
                continue
            rows.append({**base, "metric": metric, **mean_sd_ci(group[metric])})
    return pd.DataFrame(rows)


def alignment_effects(
    correlations: pd.DataFrame,
    detections: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows, det_rows = [], []
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()
    corr_index = correlations.set_index(
        ["model_id", "eval_epsilon", "subset", "score"]
    ) if not correlations.empty else None
    det_index = detections.set_index(
        ["model_id", "eval_epsilon", "subset", "score", "task", "tau"]
    ) if not detections.empty else None

    for pair in pairs.to_dict("records"):
        base = {k: pair[k] for k in [
            "comparison_id", "architecture", "family", "seed", "eval_split",
            "train_epsilon", "rob_lam", "robust_config_id",
        ]}
        if corr_index is not None:
            std = correlations[correlations["model_id"] == pair["standard_model_id"]]
            rob = correlations[correlations["model_id"] == pair["robust_model_id"]]
            merged = std.merge(
                rob, on=["eval_epsilon", "subset", "score"], suffixes=("_standard", "_robust")
            )
            for row in merged.to_dict("records"):
                delta_p = row["rho_pessimistic_robust"] - row["rho_pessimistic_standard"]
                delta_o = row["rho_optimistic_robust"] - row["rho_optimistic_standard"]
                corr_rows.append({
                    **base,
                    "eval_epsilon": row["eval_epsilon"], "subset": row["subset"],
                    "score": row["score"],
                    "rho_pessimistic_standard": row["rho_pessimistic_standard"],
                    "rho_pessimistic_robust": row["rho_pessimistic_robust"],
                    "delta_rho_pessimistic": delta_p,
                    # Scores are risk-oriented, so a more negative rho means
                    # stronger risk-versus-safety alignment.
                    "inverse_alignment_gain_pessimistic": -delta_p,
                    "rho_optimistic_standard": row["rho_optimistic_standard"],
                    "rho_optimistic_robust": row["rho_optimistic_robust"],
                    "delta_rho_optimistic": delta_o,
                    "inverse_alignment_gain_optimistic": -delta_o,
                })
        if det_index is not None:
            std = detections[detections["model_id"] == pair["standard_model_id"]]
            rob = detections[detections["model_id"] == pair["robust_model_id"]]
            merged = std.merge(
                rob,
                on=["eval_epsilon", "subset", "score", "task", "tau"],
                suffixes=("_standard", "_robust"),
            )
            for row in merged.to_dict("records"):
                det_rows.append({
                    **base,
                    "eval_epsilon": row["eval_epsilon"], "subset": row["subset"],
                    "score": row["score"], "task": row["task"], "tau": row["tau"],
                    "auroc_standard": row["auroc_standard"],
                    "auroc_robust": row["auroc_robust"],
                    "delta_auroc": row["auroc_robust"] - row["auroc_standard"],
                    "auprc_standard": row["auprc_standard"],
                    "auprc_robust": row["auprc_robust"],
                    "delta_auprc": row["auprc_robust"] - row["auprc_standard"],
                    "prevalence_standard": row["prevalence_standard"],
                    "prevalence_robust": row["prevalence_robust"],
                })
    return pd.DataFrame(corr_rows), pd.DataFrame(det_rows)


def model_level_summary(clean: pd.DataFrame, pgd: pd.DataFrame) -> pd.DataFrame:
    result = clean.copy()
    if pgd.empty or "accuracy" not in pgd:
        result["pgd_accuracy_auc"] = math.nan
        return result
    auc_rows = []
    for model_id, group in pgd.groupby("model_id", dropna=False):
        auc_rows.append({
            "model_id": model_id,
            "pgd_accuracy_auc": normalized_auc(group["eval_epsilon"], group["accuracy"]),
        })
    return result.merge(pd.DataFrame(auc_rows), on="model_id", how="left")


def tradeoff_table(models: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    if models.empty or pairs.empty or "accuracy" not in models:
        return pd.DataFrame()
    lookup = models.set_index("model_id")
    rows = []
    for pair in pairs.to_dict("records"):
        try:
            std = lookup.loc[pair["standard_model_id"]]
            rob = lookup.loc[pair["robust_model_id"]]
        except KeyError:
            continue
        rows.append({
            **{k: pair[k] for k in [
                "comparison_id", "architecture", "family", "seed", "eval_split",
                "train_epsilon", "rob_lam", "robust_config_id",
            ]},
            "clean_accuracy_standard": std["accuracy"],
            "clean_accuracy_robust": rob["accuracy"],
            "clean_accuracy_delta": rob["accuracy"] - std["accuracy"],
            "pgd_accuracy_auc_standard": std.get("pgd_accuracy_auc", math.nan),
            "pgd_accuracy_auc_robust": rob.get("pgd_accuracy_auc", math.nan),
            "pgd_accuracy_auc_delta": (
                rob.get("pgd_accuracy_auc", math.nan)
                - std.get("pgd_accuracy_auc", math.nan)
            ),
        })
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(path, index=False)
    else:
        df.sort_values(
            [c for c in ["family", "seed", "config_id", "eval_epsilon", "score"] if c in df],
            kind="stable",
        ).to_csv(path, index=False)


def family_grid(families: Sequence[str]) -> tuple[plt.Figure, np.ndarray]:
    n = len(families)
    ncols = min(2, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.6 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def config_styles(frame: pd.DataFrame) -> dict[str, tuple[Any, str]]:
    configs = list(dict.fromkeys(frame["config_id"].astype(str)))
    cmap = plt.get_cmap("tab10")
    styles = {}
    for i, config in enumerate(configs):
        color = "black" if config == "standard" else cmap(i % 10)
        styles[config] = (color, "--" if config == "standard" else "-")
    return styles


def plot_pgd_curves(pgd: pd.DataFrame, path: Path, dpi: int) -> None:
    if pgd.empty or "accuracy" not in pgd:
        return
    families = sorted(pgd["family"].unique())
    fig, axes = family_grid(families)
    styles = config_styles(pgd)
    handles: dict[str, Any] = {}
    for family, ax in zip(families, axes):
        fam = pgd[pgd["family"] == family]
        for config, group in fam.groupby("config_id", sort=False):
            color, ls = styles[config]
            for _, seed_group in group.groupby("seed"):
                seed_group = seed_group.sort_values("eval_epsilon")
                ax.plot(seed_group["eval_epsilon"], seed_group["accuracy"],
                        color=color, alpha=0.16, lw=1)
            agg = group.groupby("eval_epsilon")["accuracy"].agg(["mean", "std"]).reset_index()
            line, = ax.plot(agg["eval_epsilon"], agg["mean"], color=color, ls=ls,
                            marker="o", lw=2, label=group["config_label"].iloc[0])
            handles[group["config_label"].iloc[0]] = line
        ax.set(title=family, xlabel=r"PGD $\epsilon_{eval}$", ylabel="Robust accuracy",
               ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(3, len(handles)))
    fig.suptitle("Empirical robustness curves (thin lines are individual seeds)")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(tradeoff: pd.DataFrame, path: Path, dpi: int) -> None:
    if tradeoff.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    families = sorted(tradeoff["family"].unique())
    cmap = plt.get_cmap("tab10")
    configs = list(
        dict.fromkeys(
            (float(row.train_epsilon), float(row.rob_lam))
            for row in tradeoff[["train_epsilon", "rob_lam"]].itertuples(index=False)
        )
    )
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    marker_for = {config: markers[i % len(markers)] for i, config in enumerate(configs)}
    for i, family in enumerate(families):
        fam = tradeoff[tradeoff["family"] == family]
        for config, group in fam.groupby(["train_epsilon", "rob_lam"]):
            marker = marker_for[(float(config[0]), float(config[1]))]
            ax.scatter(group["clean_accuracy_delta"], group["pgd_accuracy_auc_delta"],
                       color=cmap(i), marker=marker, alpha=0.25, s=45)
            ax.scatter(group["clean_accuracy_delta"].mean(),
                       group["pgd_accuracy_auc_delta"].mean(),
                       color=cmap(i), marker=marker, edgecolor="black", s=110)
    ax.axhline(0, color="0.35", lw=1)
    ax.axvline(0, color="0.35", lw=1)
    ax.set(xlabel=r"$\Delta$ clean accuracy (robust - standard)",
           ylabel=r"$\Delta$ normalized PGD-accuracy AUC",
           title="Clean-utility versus empirical-robustness trade-off")
    ax.grid(alpha=0.25)
    family_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap(i),
               markeredgecolor="black", label=family, markersize=8)
        for i, family in enumerate(families)
    ]
    config_handles = [
        Line2D([0], [0], marker=marker_for[config], color="none",
               markerfacecolor="0.7", markeredgecolor="black",
               label=f"eps={config[0]:g}, lam={config[1]:g}", markersize=8)
        for config in configs
    ]
    legend_family = ax.legend(handles=family_handles, title="Family", loc="lower right")
    ax.add_artist(legend_family)
    ax.legend(handles=config_handles, title="Robust training", loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_certification(cert: pd.DataFrame, figures: Path, dpi: int) -> None:
    required = {"mean_p_safe_point", "mean_p_safe_lower", "mean_p_safe_upper",
                "mean_unknown_frac"}
    if cert.empty or not required.issubset(cert.columns):
        return
    styles = config_styles(cert)
    for family, fam in cert.groupby("family"):
        fig, (ax_p, ax_u) = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True)
        for config, group in fam.groupby("config_id", sort=False):
            color, ls = styles[config]
            label = group["config_label"].iloc[0]
            agg = group.groupby("eval_epsilon").agg(
                point=("mean_p_safe_point", "mean"),
                lower=("mean_p_safe_lower", "mean"),
                upper=("mean_p_safe_upper", "mean"),
                unknown=("mean_unknown_frac", "mean"),
            ).reset_index()
            ax_p.plot(agg["eval_epsilon"], agg["point"], color=color, ls=ls,
                      marker="o", lw=2, label=label)
            ax_p.fill_between(agg["eval_epsilon"], agg["lower"], agg["upper"],
                              color=color, alpha=0.10)
            ax_u.plot(agg["eval_epsilon"], agg["unknown"], color=color, ls=ls,
                      marker="o", lw=2, label=label)
        ax_p.set(ylabel=r"Mean $P_{safe}$", ylim=(-0.02, 1.02),
                 title=f"{family}: point estimate and mean lower/upper bounds")
        ax_u.set(xlabel=r"Certification $\epsilon_{eval}$", ylabel="Unknown fraction",
                 ylim=(-0.02, 1.02))
        for ax in (ax_p, ax_u):
            ax.grid(alpha=0.25)
        ax_p.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"certification_curves_{family}.png", dpi=dpi,
                    bbox_inches="tight")
        plt.close(fig)


def plot_detection(detection: pd.DataFrame, figures: Path, dpi: int) -> None:
    view = detection[
        (detection["task"] == "pgd_failure")
        & (detection["subset"] == "clean_correct")
    ].copy()
    if view.empty:
        return
    for family, fam in view.groupby("family"):
        configs = list(dict.fromkeys(fam["config_id"]))
        ncols = min(2, len(configs))
        nrows = int(math.ceil(len(configs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows),
                                 squeeze=False)
        for ax, config in zip(axes.flat, configs):
            group = fam[fam["config_id"] == config]
            for score, score_group in group.groupby("score"):
                agg = score_group.groupby("eval_epsilon")["auroc"].mean().reset_index()
                ax.plot(agg["eval_epsilon"], agg["auroc"], marker="o", lw=2,
                        label=score.replace("_", " "))
            ax.axhline(0.5, color="0.5", ls=":")
            ax.set(title=group["config_label"].iloc[0],
                   xlabel=r"$\epsilon_{eval}$", ylabel="AUROC for PGD failure",
                   ylim=(0, 1.02))
            ax.grid(alpha=0.25)
        for ax in axes.flat[len(configs):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
        fig.suptitle(f"{family}: uncertainty as an attack-failure detector\n(clean-correct inputs only)")
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.savefig(figures / f"pgd_detection_{family}.png", dpi=dpi,
                    bbox_inches="tight")
        plt.close(fig)


def plot_mechanism(effects: pd.DataFrame, path: Path, dpi: int) -> None:
    if effects.empty:
        return
    cert = effects[effects["outcome_group"] == "certification"]
    point = cert[cert["metric"] == "mean_p_safe_point"][
        ["comparison_id", "eval_epsilon", "delta"]
    ].rename(columns={"delta": "delta_p_safe"})
    mean_net = cert[cert["metric"] == "mean_net_frac_safe"][
        ["comparison_id", "eval_epsilon", "delta"]
    ].rename(columns={"delta": "delta_mean_net_safe"})
    frame = point.merge(mean_net, on=["comparison_id", "eval_epsilon"])
    if frame.empty:
        return
    meta = effects.drop_duplicates("comparison_id")[
        ["comparison_id", "family", "seed", "train_epsilon", "rob_lam"]
    ]
    frame = frame.merge(meta, on="comparison_id")
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (family, group) in enumerate(frame.groupby("family")):
        ax.scatter(group["delta_mean_net_safe"], group["delta_p_safe"],
                   alpha=0.55, label=family, color=plt.get_cmap("tab10")(i))
    lo = min(frame["delta_mean_net_safe"].min(), frame["delta_p_safe"].min(), 0)
    hi = max(frame["delta_mean_net_safe"].max(), frame["delta_p_safe"].max(), 0)
    ax.plot([lo, hi], [lo, hi], color="0.45", ls=":", label="equal change")
    ax.axhline(0, color="0.75", lw=1)
    ax.axvline(0, color="0.75", lw=1)
    ax.set(xlabel=r"$\Delta$ mean-network safe fraction",
           ylabel=r"$\Delta$ mean pessimistic $P_{safe}$ point estimate",
           title="Mean-network movement versus posterior robust-mass change")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 3) -> str:
    if frame.empty:
        return "_No applicable rows._"
    view = frame.loc[:, [c for c in columns if c in frame]].copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(
            lambda x: "" if not np.isfinite(x) else f"{x:.{digits}f}"
        )
    header = "| " + " | ".join(view.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def write_report(
    path: Path,
    inventory: pd.DataFrame,
    tradeoff: pd.DataFrame,
    effects_summary: pd.DataFrame,
    corr_effects: pd.DataFrame,
    det_effects: pd.DataFrame,
    checks: pd.DataFrame,
    missing_margin: bool,
) -> None:
    trade_summary = pd.DataFrame()
    if not tradeoff.empty:
        rows = []
        keys = ["family", "train_epsilon", "rob_lam"]
        for group_keys, group in tradeoff.groupby(keys, dropna=False):
            rows.append({
                **dict(zip(keys, group_keys)),
                "n_seeds": group["seed"].nunique(),
                "clean_delta": group["clean_accuracy_delta"].mean(),
                "pgd_auc_delta": group["pgd_accuracy_auc_delta"].mean(),
            })
        trade_summary = pd.DataFrame(rows)

    cert_effect = effects_summary[
        (effects_summary.get("outcome_group") == "certification")
        & (effects_summary.get("metric") == "mean_p_safe_point")
    ] if not effects_summary.empty else pd.DataFrame()

    corr_summary = pd.DataFrame()
    if not corr_effects.empty:
        core = corr_effects[
            (corr_effects["subset"] == "clean_correct")
            & corr_effects["score"].isin(["mutual_information", "one_minus_confidence"])
        ]
        rows = []
        keys = ["family", "train_epsilon", "rob_lam", "eval_epsilon", "score", "subset"]
        for group_keys, group in core.groupby(keys, dropna=False):
            rows.append({**dict(zip(keys, group_keys)),
                         **mean_sd_ci(group["inverse_alignment_gain_pessimistic"])})
        corr_summary = pd.DataFrame(rows)

    det_summary = pd.DataFrame()
    if not det_effects.empty:
        view = det_effects[
            (det_effects["task"] == "pgd_failure")
            & (det_effects["subset"] == "clean_correct")
            & det_effects["score"].isin(["mutual_information", "one_minus_confidence"])
        ]
        rows = []
        keys = ["family", "train_epsilon", "rob_lam", "eval_epsilon", "score"]
        for group_keys, group in view.groupby(keys, dropna=False):
            rows.append({**dict(zip(keys, group_keys)), **mean_sd_ci(group["delta_auroc"])})
        det_summary = pd.DataFrame(rows)

    errors = int((checks.get("level", pd.Series(dtype=str)) == "error").sum())
    warnings = int((checks.get("level", pd.Series(dtype=str)) == "warning").sum())
    margin_note = (
        "- Predictive margin was not present in the JSON. The script used entropy, "
        "mutual information, expected entropy, and 1-confidence. Add the top-two "
        "probability margin to future experiment payloads; existing runs cannot recover it "
        "without checkpoint inference.\n"
        if missing_margin else "- Predictive margin was available and analyzed as 1-margin.\n"
    )
    text = f"""# Adversarial experiment analysis

Loaded **{len(inventory)} paired JSON files** covering **{inventory['family'].nunique()} model families** and **{inventory['seed'].nunique()} seeds**.

## Interpretation rules

- PGD accuracy concerns the posterior-predictive classifier and is diagnostic, not a certificate.
- `p_safe_point` is the pessimistic ranking estimate (unknown counted unsafe).
- Lower/upper robustness bounds support threshold claims; inconclusive cases remain unknown.
- Alignment metrics were calculated within each seed. Seed-input rows were not pooled.
- All deltas are robust-training minus standard-training unless explicitly named an inverse-alignment gain.
{margin_note}
## Clean/PGD trade-off

{markdown_table(trade_summary, ['family', 'train_epsilon', 'rob_lam', 'n_seeds', 'clean_delta', 'pgd_auc_delta'])}

## Posterior robust-mass effect

{markdown_table(cert_effect, ['family', 'train_epsilon', 'rob_lam', 'eval_epsilon', 'n_seeds', 'mean', 'sd', 'ci_low', 'ci_high'])}

Here `mean` is the paired change in the mean pessimistic `p_safe_point` across seeds.

## Inverse uncertainty-safety alignment effect

{markdown_table(corr_summary, ['family', 'train_epsilon', 'rob_lam', 'eval_epsilon', 'score', 'subset', 'n_seeds', 'mean', 'sd'])}

A positive inverse-alignment gain means the risk score became more negatively associated with safe posterior mass after robust training.

## PGD-failure detection effect

{markdown_table(det_summary, ['family', 'train_epsilon', 'rob_lam', 'eval_epsilon', 'score', 'n_seeds', 'mean', 'sd'])}

Here `mean` is the paired AUROC change on clean-correct inputs. Always interpret it beside failure prevalence and robust accuracy: discrimination may fall when nearly every input belongs to one class.

## Quality control

- Errors: **{errors}**
- Warnings: **{warnings}**

See `tables/quality_checks.csv` before interpreting the results. A failed or inconclusive verification is not coded as non-robust.
"""
    path.write_text(text, encoding="utf-8")


def analyze(args: argparse.Namespace) -> Path:
    runs_dir = args.runs_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    taus = sorted(set(float(v) for v in args.tau))
    if any(not 0 < tau < 1 for tau in taus):
        raise ValueError("Every --tau value must lie strictly between 0 and 1")

    data = load_runs(runs_dir, taus)
    frames = {
        "run_inventory": pd.DataFrame(data.inventory),
        "clean_metrics": pd.DataFrame(data.clean),
        "pgd_metrics": pd.DataFrame(data.pgd),
        "certification_metrics": pd.DataFrame(data.certification),
        "per_input": pd.DataFrame(data.per_input),
        "posterior_spread": pd.DataFrame(data.posterior_spread),
        "paired_effects": pd.DataFrame(data.paired_effects),
        "posterior_spread_effects": pd.DataFrame(data.spread_effects),
        "pairs": pd.DataFrame(data.pairs),
    }
    if frames["per_input"].empty:
        raise ValueError("No per-input robustness data could be parsed")

    correlations, detections, outcomes = alignment_tables(frames["per_input"], taus)
    corr_effects, det_effects = alignment_effects(correlations, detections, frames["pairs"])
    frames.update({
        "alignment_correlations": correlations,
        "detection_metrics": detections,
        "certification_outcomes": outcomes,
        "certificate_diagnostics": certificate_diagnostics(frames["per_input"]),
        "alignment_correlation_effects": corr_effects,
        "detection_effects": det_effects,
        "paired_effect_summary": effect_summary(frames["paired_effects"]),
        "correlation_seed_summary": seed_summary(
            correlations, ["rho_pessimistic", "rho_optimistic"], ["subset", "score"]
        ),
        "detection_seed_summary": seed_summary(
            detections, ["auroc", "auprc", "prevalence"], ["subset", "score", "task", "tau"]
        ),
    })
    frames["model_level_summary"] = model_level_summary(
        frames["clean_metrics"], frames["pgd_metrics"]
    )
    frames["robustness_tradeoff"] = tradeoff_table(
        frames["model_level_summary"], frames["pairs"]
    )

    check_monotonicity(data, frames["pgd_metrics"], frames["certification_metrics"])
    missing_margin = frames["per_input"]["predictive_margin"].notna().sum() == 0
    if missing_margin:
        data.check("warning", str(runs_dir), "predictive_margin_missing",
                   "top-two predictive probability margin is absent from all payloads")
    frames["quality_checks"] = pd.DataFrame(data.checks)

    for name, frame in frames.items():
        write_csv(frame, tables / f"{name}.csv")

    if not args.no_plots:
        plot_pgd_curves(frames["pgd_metrics"], figures / "pgd_accuracy_curves.png", args.dpi)
        plot_tradeoff(frames["robustness_tradeoff"], figures / "clean_pgd_tradeoff.png", args.dpi)
        plot_certification(frames["certification_metrics"], figures, args.dpi)
        plot_detection(detections, figures, args.dpi)
        plot_mechanism(frames["paired_effects"], figures / "mechanism_deltas.png", args.dpi)

    write_report(
        out_dir / "analysis_report.md",
        frames["run_inventory"], frames["robustness_tradeoff"],
        frames["paired_effect_summary"], corr_effects, det_effects,
        frames["quality_checks"], missing_margin,
    )

    if args.strict and not frames["quality_checks"].empty:
        n_errors = int((frames["quality_checks"]["level"] == "error").sum())
        if n_errors:
            raise RuntimeError(f"Analysis completed with {n_errors} quality-control errors")
    return out_dir


def synthetic_payload(seed: int, family: str, train_eps: float, lam: float) -> dict[str, Any]:
    """Small deterministic fixture used by --self-test."""
    rng = np.random.default_rng(seed + (0 if family == "bbb" else 100))
    n, cert_eps, pgd_eps = 20, [0.01, 0.03], [0.01, 0.03, 0.05]

    def side(robust: bool) -> dict[str, Any]:
        shift = 0.12 if robust else 0.0
        latent = np.clip(rng.beta(3, 2, n) + shift, 0, 0.82)
        unknown = np.full(n, 0.10 if robust else 0.16)
        entropy = np.clip(1.1 - latent + rng.normal(0, 0.05, n), 0, None)
        confidence = np.clip(latent + 0.1, 0, 1)
        correct = (confidence > 0.35).astype(int)
        report: dict[str, Any] = {
            "clean": {"accuracy": float(correct.mean()), "nll": float(0.7 - shift),
                      "brier": float(0.25 - shift / 2), "ece": float(0.08)},
            "cert_subset_uncertainty": {
                "n_mc_samples": 20,
                "predictive_entropy": entropy.tolist(),
                "mutual_information": (entropy * 0.25).tolist(),
                "expected_entropy": (entropy * 0.75).tolist(),
                "confidence": confidence.tolist(),
                "correct": correct.tolist(),
            },
            "pgd": {}, "cert_subset_pgd_hit": {}, "certification": {},
        }
        for eps in pgd_eps:
            acc = np.clip(correct.mean() - eps * (5.5 - 2 * robust), 0, 1)
            report["pgd"][f"{eps:g}"] = {"accuracy": float(acc), "nll": float(1 - acc)}
        for eps in cert_eps:
            p = np.clip(latent - eps * 4, 0, 1)
            p_opt = p + unknown
            lower = np.clip(p - 0.08, 0, 1)
            upper = np.clip(p_opt + 0.08, 0, 0.925)
            hit = (latent < (0.52 + eps * 2 - shift)).astype(int)
            report["cert_subset_pgd_hit"][f"{eps:g}"] = hit.tolist()
            per = {
                "true_label": (np.arange(n) % 10).tolist(),
                "p_safe_lower": lower.tolist(), "p_safe_upper": upper.tolist(),
                "p_safe_point": p.tolist(), "p_safe_point_optimistic": p_opt.tolist(),
                "p_safe_ci_low": np.clip(p - 0.04, 0, 1).tolist(),
                "p_safe_ci_high": np.clip(p + 0.04, 0, 1).tolist(),
                "unknown_frac": unknown.tolist(), "n_samples_used": [292] * n,
                "n_property_evaluations": [292] * n,
                "mean_net_verdict": ["safe" if v > 0.5 else "unsafe" for v in p],
            }
            report["certification"][f"{eps:g}"] = {
                "n_inputs": n, "mean_p_safe_lower": float(lower.mean()),
                "mean_p_safe_upper": float(upper.mean()),
                "mean_interval_width": float((upper - lower).mean()),
                "frac_p_safe_lower_geq_0p5": float((lower >= 0.5).mean()),
                "mean_p_safe_point": float(p.mean()),
                "median_p_safe_point": float(np.median(p)),
                "frac_p_safe_point_geq_0p9": float((p >= 0.9).mean()),
                "mean_unknown_frac": float(unknown.mean()),
                "mean_property_evaluations": 292.0,
                "mean_net_frac_safe": float((p > 0.5).mean()),
                "mean_net_frac_unsafe": float((p <= 0.5).mean()),
                "per_input": per,
            }
        return report

    standard, robust = side(False), side(True)
    layers = [{"layer": 0, "sigma_w_mean": 0.04, "sigma_w_median": 0.03,
               "sigma_w_max": 0.12, "sigma_b_mean": 0.03,
               "relative_spread_mean": 0.2}]
    return {
        "experiment": {"dataset": "MNIST", "architecture": "mlp128x1",
                       "family": family, "seed": seed, "eval_split": "val",
                       "eval_inputs": 2000, "certification_inputs": n,
                       "mc_samples": 20, "pgd_enabled": 1,
                       "certification_enabled": 1},
        "configuration": {
            "cert": {"theta": 0.075, "gamma": 0.075, "alpha": 0.05},
            "robust_point": {"epsilon": train_eps, "rob_lam": lam},
        },
        "posterior_sigma": {
            "standard": {"kind": "gaussian", "layers": layers},
            "robust": {"kind": "gaussian", "layers": [
                {**layers[0], "sigma_w_mean": 0.032}
            ]},
        },
        "standard": standard, "robust": robust,
        "comparison": {"clean_accuracy_delta": robust["clean"]["accuracy"]
                       - standard["clean"]["accuracy"]},
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bnn-analysis-test-") as tmp:
        root = Path(tmp)
        runs, out = root / "runs", root / "analysis"
        runs.mkdir()
        for family in ("bbb", "vogn"):
            for seed in range(3):
                payload = synthetic_payload(seed, family, 0.03, 0.75)
                (runs / f"{family}_seed{seed}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
        args = argparse.Namespace(
            runs_dir=runs, out_dir=out, tau=[0.5, 0.9], dpi=80,
            no_plots=False, strict=True,
        )
        analyze(args)
        required = [
            out / "analysis_report.md",
            out / "tables" / "alignment_correlations.csv",
            out / "tables" / "detection_effects.csv",
            out / "figures" / "pgd_accuracy_curves.png",
        ]
        missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
        if missing:
            raise AssertionError(f"Self-test outputs missing: {missing}")
    print("Self-test passed.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze adversarial_run.py JSON outputs without pooling seeds."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                        help="Root containing experiment JSON files (recursive).")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/adversarial"),
                        help="Destination for tables, figures, and analysis_report.md.")
    parser.add_argument("--tau", type=float, action="append", default=None,
                        help="Posterior-safe-mass decision threshold; repeatable. "
                             "Defaults to 0.5 and 0.9.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero when quality-control errors are found.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run an end-to-end synthetic-data test and exit.")
    args = parser.parse_args(argv)
    if args.tau is None:
        args.tau = [0.5, 0.9]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    try:
        out_dir = analyze(args)
    except Exception as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Analysis complete: {out_dir}")


if __name__ == "__main__":
    main()
