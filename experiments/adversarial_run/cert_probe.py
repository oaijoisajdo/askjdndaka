"""
Statistical P_safe probe per input per eps, plus the sigma -> 0 control.

Serializes per-input arrays: the aggregates alone cannot feed the Phase C
rank-correlation / ROC analysis, and the delta tables in v1 were censored by
the theta floor/ceiling.
"""

from __future__ import annotations

import jax
import numpy as np

from certification import (
    estimate_probabilistic_robustness_bounds,
    make_phi2,
    propagate_deterministic_LBP,
)
from experiments.config import CONFIG
from experiments.posterior_stats import mean_network_layers
from experiments.utils import epsilon_key

# Sub-stream for the zero-spread control, so the mean-network verdict never
# consumes randomness from the estimator's stream.
_MEAN_NET_FOLD = 999_983


def _probe_one_input(posterior, x_i, y_i, *, eps, key, mean_layers,
                     pgd_candidates) -> dict:
    """One (input, radius) cell of the probe. Returns one flat row."""
    c = CONFIG.cert
    x_star = np.asarray(x_i)[None, :]

    phi = make_phi2(
        x_star=x_star,
        x_L=np.clip(x_star - eps, 0.0, 1.0),
        x_U=np.clip(x_star + eps, 0.0, 1.0),
        y_dim=CONFIG.n_classes,
        class_selection="true_label",
        true_label=int(y_i),
        propagate_fn=propagate_deterministic_LBP,
        ibp_first=c.ibp_first,
        candidates_fn=pgd_candidates,
    )

    bounds = estimate_probabilistic_robustness_bounds(
        posterior=posterior,
        property_fn=phi,
        key=key,
        theta=c.theta,
        gamma=c.gamma,
        alpha=c.alpha,
    )
    pess = bounds.unknown_as_violation   # unknown counted as unsafe
    opt = bounds.unknown_as_safe         # unknown counted as safe

    return {
        "true_label": int(y_i),
        # theta-shifted Massart bounds (guarantees; censored at 0/0.925)
        "p_safe_lower": float(bounds.robustness_lower),
        "p_safe_upper": float(bounds.robustness_upper),
        # un-censored point estimates (use these for ranking/correlation)
        "p_safe_point": float(1.0 - pess.p_hat),
        "p_safe_point_optimistic": float(1.0 - opt.p_hat),
        # CP interval on violation prob -> flip to robustness
        "p_safe_ci_low": float(1.0 - pess.ci_high),
        "p_safe_ci_high": float(1.0 - pess.ci_low),
        # verifier diagnostics
        "unknown_frac": float(pess.unk / pess.n),
        "n_samples_used": int(pess.n),
        "n_property_evaluations": int(bounds.n_property_evaluations),
        # sigma -> 0 control: verdict for the posterior-mean network
        "mean_net_verdict": (
            phi(mean_layers, jax.random.fold_in(key, _MEAN_NET_FOLD))
            if mean_layers is not None else None
        ),
    }


def _columns(rows: list[dict]) -> dict[str, list]:
    """Row-major probe output -> the column-major layout the JSON expects."""
    return {name: [row[name] for row in rows] for name in rows[0]}


def _aggregate(cols: dict[str, list], *, n_inputs: int, has_mean_net: bool) -> dict:
    p_lo = np.asarray(cols["p_safe_lower"])
    p_hi = np.asarray(cols["p_safe_upper"])
    p_pt = np.asarray(cols["p_safe_point"])
    verdicts = cols["mean_net_verdict"]

    def frac_verdict(target):
        return float(np.mean([v == target for v in verdicts])) if has_mean_net else None

    return {
        "n_inputs": n_inputs,
        # Bound-based aggregates (comparable to v1, still censored)
        "mean_p_safe_lower": float(p_lo.mean()),
        "mean_p_safe_upper": float(p_hi.mean()),
        "mean_interval_width": float((p_hi - p_lo).mean()),
        "frac_p_safe_lower_geq_0p5": float(np.mean(p_lo >= 0.5)),
        # Point-estimate aggregates (uncensored; report these for deltas)
        "mean_p_safe_point": float(p_pt.mean()),
        "median_p_safe_point": float(np.median(p_pt)),
        "frac_p_safe_point_geq_0p9": float(np.mean(p_pt >= 0.9)),
        # Verifier diagnostics
        "mean_unknown_frac": float(np.mean(cols["unknown_frac"])),
        "mean_property_evaluations": float(np.mean(cols["n_property_evaluations"])),
        # Zero-spread control aggregate (None if unavailable)
        "mean_net_frac_safe": frac_verdict("safe"),
        "mean_net_frac_unsafe": frac_verdict("unsafe"),
        "per_input": cols,
    }


def cert_probe(posterior, x, y, *, seed: int, pgd_candidates) -> dict:
    mean_layers = mean_network_layers(posterior)
    base_key = jax.random.key(seed)

    results = {}
    for eps in CONFIG.cert.eps_values:
        rows = [
            _probe_one_input(
                posterior, x[i], y[i],
                eps=eps,
                key=jax.random.fold_in(base_key, i),
                mean_layers=mean_layers,
                pgd_candidates=pgd_candidates,
            )
            for i in range(len(x))
        ]
        results[epsilon_key(eps)] = _aggregate(
            _columns(rows),
            n_inputs=len(x),
            has_mean_net=mean_layers is not None,
        )

    return results
