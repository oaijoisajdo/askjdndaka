"""
Within-seed reduction: inputs -> one statistic per (run, epsilon).

This is the layer the plan's core rule governs. Every correlation, AUROC,
ECE and risk-coverage curve is computed here, inside a single seed, before
anything is averaged. Nothing in this module pools rows across seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import alignment as _align
from analysis import metrics as M
from analysis.loading import RUN_KEYS
from analysis.scores import DEFAULT_SCORES, VARIANTS

# Certification thresholds for the robust / inconclusive / below-threshold
# split. Applied to the BOUNDS, not the point estimate: a bound is a
# guarantee, and mixing the two would report a point estimate as certified.
DEFAULT_TAUS = (0.5, 0.9, 0.95)

DEFAULT_COVERAGES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

_GROUP = [*RUN_KEYS, "eps"]


def _group_keys(g: pd.DataFrame) -> dict:
    return {k: g[k].iloc[0] for k in _GROUP}


# ---------------------------------------------------------------------------
# A + C: subset-level outcome rates
# ---------------------------------------------------------------------------

def subset_outcomes(per_input: pd.DataFrame) -> pd.DataFrame:
    """
    Accuracy and attack-success rates on the certification subset.

    These are subset quantities, not population ones: the subset is
    class-balanced by construction, so its accuracy is a balanced accuracy and
    will not generally match the eval-split figure. Kept separate from
    ``eval_metrics`` for exactly that reason -- do not plot them on one axis.
    """
    rows = []
    for _, g in per_input.groupby(_GROUP, dropna=False):
        rows.append({
            **_group_keys(g),
            "n_inputs": len(g),
            "subset_accuracy_clean": g["correct_clean"].mean(),
            "subset_accuracy_adv": g["correct_adv"].mean(),
            "subset_misclassification_adv": g["adv_error"].mean(),
            # Conditional on having been correct before the attack.
            "attack_success_rate": g["attack_success"].mean(),
            "n_clean_correct": int(g["correct_clean"].sum()),
            # Calibration on the subset. The plan wants ECE on the full eval
            # split; this is the subset stand-in and is labelled as such.
            "subset_ece_clean": M.ece(1.0 - g["one_minus_confidence_clean"],
                                      g["correct_clean"]),
            "subset_ece_adv": M.ece(1.0 - g["one_minus_confidence_adv"],
                                    g["correct_adv"]),
        })
    return pd.DataFrame(rows)


def mean_uncertainty(per_input: pd.DataFrame, scores=DEFAULT_SCORES) -> pd.DataFrame:
    """
    Mean U_clean, U_adv and Delta U, plus Delta U stratified by attack success.

    The stratification is the substantive part: a mean Delta U that rises with
    epsilon says little if it rises equally on inputs the attack failed to
    flip. The gap between the two strata is what "uncertainty responds to
    adversarial shift" should mean.
    """
    rows = []
    for _, g in per_input.groupby(_GROUP, dropna=False):
        row = _group_keys(g)
        hit = g["attack_success"] == 1
        miss = g["attack_success"] == 0
        for score in scores:
            for variant in VARIANTS:
                row[f"mean_{score}_{variant}"] = g[f"{score}_{variant}"].mean()
            d = f"{score}_delta"
            row[f"delta_{score}_attack_success"] = g.loc[hit, d].mean()
            row[f"delta_{score}_attack_failed"] = g.loc[miss, d].mean()
            row[f"delta_{score}_success_gap"] = (
                g.loc[hit, d].mean() - g.loc[miss, d].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# C + E: detection and alignment, long format
# ---------------------------------------------------------------------------

def _cert_labels(g: pd.DataFrame, tau: float):
    """
    Three-way certification verdict at tau, DIRECTION-VALID.

    robust:          Clopper-Pearson lower limit >= tau   (certified safe)
    below_threshold: theta-margin upper bound    <  tau   (certified unsafe)
    inconclusive:    everything else

    Delegates to ``analysis.alignment.add_labels`` so this module and the
    alignment analysis cannot drift apart: one definition, two callers.

    Two changes from the previous version, both recorded in the frozen
    protocol:

    1. The robust direction is gated on ``p_safe_ci_low`` (Clopper-Pearson),
       not on ``p_safe_lower`` (theta-adjusted Massart). Subtracting the
       Massart slack from a quantity that already needs a high-confidence
       floor is conservative twice over. At tau = 0.9 this moves order 1% of
       rows across the boundary.
    2. Inconclusive rows are no longer dropped. Exclusion is not neutral:
       inconclusiveness is itself predictable from the uncertainty scores
       (measured AUROC up to ~0.95 at moderate epsilon), and it varies by
       family and radius, so dropping those rows removes exactly the inputs
       where the score is most informative, differentially across cells.
       Callers receive the mask and both conservative assignments instead;
       see ``_cert_label_bracket``.

    Returned for backward compatibility: (conclusive_mask, failure_label).
    """
    d = _align.add_labels(g, tau)
    return ~d["lab_inconclusive"], d["lab_below"].astype(float)


def _cert_label_bracket(g: pd.DataFrame, tau: float):
    """
    Conservative-assignment bracket for the failure label.

    Returns ``(fail_as_below, fail_as_robust, inconclusive_mask)`` over ALL
    rows, where the two failure vectors resolve every inconclusive row to
    below-threshold and to robust respectively. Any metric computed on both
    is bracketed: the true value under any resolution of the verifier's
    unknowns lies between them, and no rows are discarded.
    """
    d = _align.add_labels(g, tau)
    inc = d["lab_inconclusive"]
    return ((d["lab_below"] | inc).astype(float),
            d["lab_below"].astype(float),
            inc)


def detection_and_alignment(
    per_input: pd.DataFrame,
    scores=DEFAULT_SCORES,
    taus=DEFAULT_TAUS,
) -> pd.DataFrame:
    """
    Long table keyed by (run, eps, score, variant, target).

    Targets:
      adv_error  -- analysis C: does the score rank adversarial errors first?
      cert_tau=* -- analysis E: does it rank certification failures first?

    Spearman rho against P_safe is attached to the adv_error rows so the
    alignment correlation and the detection metrics live on one row set;
    it is NaN wherever the verifier did not run at that epsilon.
    """
    rows = []
    for _, g in per_input.groupby(_GROUP, dropna=False):
        keys = _group_keys(g)
        for score in scores:
            for variant in VARIANTS:
                s = g[f"{score}_{variant}"]

                # Flagged rather than inferred: a NaN AUROC can mean "no
                # positives at this epsilon" or "the score is degenerate for
                # this family", and the two must not be pooled.
                constant = M.is_constant(s)

                rows.append({
                    **keys, "score": score, "variant": variant,
                    "target": "adv_error",
                    "score_constant": constant,
                    "auroc": M.auroc(s, g["adv_error"]),
                    "auprc": M.auprc(s, g["adv_error"]),
                    "prevalence": M.prevalence(g["adv_error"]),
                    "n_used": int(g["adv_error"].notna().sum()),
                    # Higher is better, per the plan's "plot -rho" convention:
                    # risk should fall as certified safe mass rises.
                    "neg_spearman_p_safe": -M.spearman(s, g["p_safe_point"]),
                    "neg_spearman_p_safe_optimistic":
                        -M.spearman(s, g["p_safe_point_optimistic"]),
                })

                for tau in taus:
                    conclusive, fail = _cert_labels(g, tau)
                    fail_inc_below, fail_inc_robust, inc = \
                        _cert_label_bracket(g, tau)
                    sc, fc = s[conclusive], fail[conclusive]
                    rows.append({
                        **keys, "score": score, "variant": variant,
                        "target": f"cert_fail_tau{tau:g}",
                        "score_constant": M.is_constant(sc),
                        # Conclusive-only: comparable to the previous build,
                        # but no longer the primary number.
                        "auroc": M.auroc(sc, fc),
                        "auprc": M.auprc(sc, fc),
                        "prevalence": M.prevalence(fc),
                        # Bracket over all rows under both conservative
                        # resolutions of the verifier's unknowns. The true
                        # AUROC lies between these two regardless of how the
                        # inconclusive rows would have resolved.
                        "auroc_inc_as_below": M.auroc(s, fail_inc_below),
                        "auroc_inc_as_robust": M.auroc(s, fail_inc_robust),
                        # Diagnostic: is inconclusiveness itself predictable?
                        # If this is far from 0.5, exclusion is not neutral.
                        "auroc_inconclusive": M.auroc(s, inc.astype(float)),
                        "n_used": int(conclusive.sum()),
                        "n_conclusive": int(conclusive.sum()),
                        "n_inconclusive": int(inc.sum()),
                        "frac_inconclusive": float(inc.mean()),
                        "neg_spearman_p_safe": np.nan,
                        "neg_spearman_p_safe_optimistic": np.nan,
                    })
    return pd.DataFrame(rows)


def risk_coverage_table(
    per_input: pd.DataFrame,
    scores=DEFAULT_SCORES,
    coverages=DEFAULT_COVERAGES,
) -> pd.DataFrame:
    """Selective risk vs coverage for adversarial-error abstention."""
    rows = []
    for _, g in per_input.groupby(_GROUP, dropna=False):
        keys = _group_keys(g)
        for score in scores:
            for variant in VARIANTS:
                for point in M.risk_coverage(
                    g[f"{score}_{variant}"], g["adv_error"], coverages
                ):
                    rows.append({**keys, "score": score,
                                 "variant": variant, **point})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D: certification, recomputed per input rather than read from the aggregates
# ---------------------------------------------------------------------------

def certification_summary(per_input: pd.DataFrame, taus=DEFAULT_TAUS) -> pd.DataFrame:
    """
    Per-seed certification statistics, including the tau splits the run
    script cannot produce (it has no tau grid) and distribution quantiles.

    Rows where the verifier did not run are dropped, so this table is defined
    only on the certification epsilon grid.
    """
    rows = []
    df = per_input[per_input["p_safe_point"].notna()]
    for _, g in df.groupby(_GROUP, dropna=False):
        row = {
            **_group_keys(g),
            "n_inputs": len(g),
            "mean_p_safe_point": g["p_safe_point"].mean(),
            "median_p_safe_point": g["p_safe_point"].median(),
            "mean_p_safe_optimistic": g["p_safe_point_optimistic"].mean(),
            "mean_p_safe_lower": g["p_safe_lower"].mean(),
            "mean_p_safe_upper": g["p_safe_upper"].mean(),
            "mean_interval_width": (g["p_safe_upper"] - g["p_safe_lower"]).mean(),
            "mean_ci_width": (g["p_safe_ci_high"] - g["p_safe_ci_low"]).mean(),
            "mean_unknown_frac": g["unknown_frac"].mean(),
            "mean_property_evaluations": g["n_property_evaluations"].mean(),
            "mean_net_frac_safe": (g["mean_net_verdict"] == "safe").mean()
            if g["mean_net_verdict"].notna().any() else np.nan,
            "mean_net_frac_unsafe": (g["mean_net_verdict"] == "unsafe").mean()
            if g["mean_net_verdict"].notna().any() else np.nan,
        }
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            row[f"p_safe_q{int(q * 100):02d}"] = g["p_safe_point"].quantile(q)
        for tau in taus:
            robust = (g["p_safe_lower"] >= tau).mean()
            below = (g["p_safe_upper"] < tau).mean()
            row[f"frac_robust_tau{tau:g}"] = robust
            row[f"frac_below_tau{tau:g}"] = below
            row[f"frac_inconclusive_tau{tau:g}"] = 1.0 - robust - below
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A: epsilon-integrated curve summaries
# ---------------------------------------------------------------------------

def curve_auc(eval_metrics: pd.DataFrame, subset: pd.DataFrame) -> pd.DataFrame:
    """
    Normalized area under each metric's epsilon curve, one value per run.

    Integrating over epsilon collapses the curve to a single number that can
    carry a seed SE and enter a paired test, which the per-epsilon curves
    cannot do without a multiple-comparison argument.
    """
    rows: dict[tuple, dict] = {}
    specs = [
        (eval_metrics, "accuracy", "pgd_accuracy_auc"),
        (eval_metrics, "nll", "nll_auc"),
        (eval_metrics, "brier", "brier_auc"),
        (eval_metrics, "ece", "ece_auc"),
        (subset, "subset_accuracy_adv", "subset_accuracy_auc"),
        (subset, "attack_success_rate", "attack_success_auc"),
    ]
    for df, column, name in specs:
        if df.empty or column not in df.columns:
            continue
        for keys, g in df.groupby(RUN_KEYS, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            # Accumulate into one row per run. NOT pivot_table: train_eps and
            # rob_lam are NaN for standard runs, and pivot_table expands the
            # cartesian product over index levels containing NaN, which
            # silently quadruples the row count with empty rows.
            #
            # NaN is also unusable as a dict key -- each groupby produces a
            # distinct NaN object and `nan != nan`, so standard runs coming
            # from two source frames would land in two separate rows. Map NaN
            # to None for lookup, then write the original values into the row.
            lookup = tuple(None if isinstance(k, float) and np.isnan(k) else k
                           for k in keys)
            row = rows.setdefault(lookup, dict(zip(RUN_KEYS, keys)))
            row[name] = M.normalized_auc(g["eps"], g[column])

    if not rows:
        return pd.DataFrame(columns=RUN_KEYS)
    return pd.DataFrame(list(rows.values()))
