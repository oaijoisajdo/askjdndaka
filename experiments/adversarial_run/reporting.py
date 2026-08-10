"""Per-model evaluation bundle and the standard-vs-robust comparison."""

from __future__ import annotations

import jax
import numpy as np

from certification import Posterior
from experiments.attack_eval import evaluate, evaluate_pgd, pgd_examples
from experiments.cert_probe import cert_probe
from experiments.config import CONFIG
from experiments.uncertainty import mc_probs_from_posterior, per_input_uncertainty
from experiments.utils import epsilon_key

# Offsets carving disjoint streams out of a single run seed.
_CERT_EVAL_OFFSET = 7      # predictive draws on the certification subset
_CERT_ATTACK_OFFSET = 1    # attack on the certification subset


def _cert_uncertainty(posterior, x, y, *, key) -> dict:
    probs = mc_probs_from_posterior(
        posterior, x, n_samples=CONFIG.mc_samples, key=key,
    )
    return per_input_uncertainty(probs, y)


def _check_eps_grids() -> tuple[dict[str, float], set[str]]:
    """Certification radii must be a subset of the PGD grid, by key."""
    pgd_eps = {epsilon_key(e): e for e in CONFIG.attack.pgd_eps}
    cert_eps = {epsilon_key(e) for e in CONFIG.cert.eps_values}
    missing = cert_eps - set(pgd_eps)
    if missing:
        raise ValueError(
            "Certification epsilons missing from the PGD grid: "
            f"{sorted(missing, key=float)}"
        )
    return pgd_eps, cert_eps


def _pgd_section(model, posterior, *, x_eval, y_eval, x_cert, y_cert,
                 seed, cert_eval_key) -> tuple[dict, dict]:
    pgd_eps, cert_eps = _check_eps_grids()
    pgd, hits = {}, {}

    for eps_key, eps in pgd_eps.items():
        # Distinct attack stream for the certification subset; the predictive
        # draws reuse cert_eval_key so that U(x_adv) - U(x) shares posterior
        # samples with the clean measurement.
        x_adv_cert = pgd_examples(
            model, x_cert, y_cert,
            eps=eps, attack_seed=seed + _CERT_ATTACK_OFFSET,
        )
        adv_uncertainty = _cert_uncertainty(
            posterior, x_adv_cert, y_cert, key=cert_eval_key,
        )

        report = evaluate_pgd(model, x_eval, y_eval, eps=eps, seed=seed)
        report["cert_subset_uncertainty"] = adv_uncertainty
        pgd[eps_key] = report

        # DEPRECATED: exactly 1 - cert_subset_uncertainty["correct"].
        # Kept only so v1 analysis scripts keep parsing; drop once migrated.
        if eps_key in cert_eps:
            correct = np.asarray(adv_uncertainty["correct"], dtype=bool)
            hits[eps_key] = (~correct).astype(int).tolist()

    return pgd, hits


def full_report(model, *, x_eval, y_eval, x_cert, y_cert, seed,
                pgd, cert, pgd_candidates) -> dict:
    posterior = Posterior(model)
    cert_eval_key = jax.random.key(
        seed + CONFIG.attack.eval_seed_offset + _CERT_EVAL_OFFSET
    )

    report = {
        "clean": evaluate(model, x_eval, y_eval, seed=seed),
        "cert_subset_uncertainty": _cert_uncertainty(
            posterior, x_cert, y_cert, key=cert_eval_key,
        ),
    }

    if pgd:
        report["pgd"], report["cert_subset_pgd_hit"] = _pgd_section(
            model, posterior,
            x_eval=x_eval, y_eval=y_eval,
            x_cert=x_cert, y_cert=y_cert,
            seed=seed, cert_eval_key=cert_eval_key,
        )

    if cert:
        report["certification"] = cert_probe(
            posterior, x_cert, y_cert,
            seed=seed, pgd_candidates=pgd_candidates,
        )

    return report


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _delta_map(standard, robust, section: str, field: str) -> dict:
    """robust - standard for one field, keyed by eps. None-safe."""
    out = {}
    for eps, s in standard[section].items():
        s_val, r_val = s[field], robust[section][eps][field]
        out[eps] = None if (s_val is None or r_val is None) else float(r_val - s_val)
    return out


def compare_reports(standard, robust) -> dict:
    comparison = {
        "clean_accuracy_delta": float(
            robust["clean"]["accuracy"] - standard["clean"]["accuracy"]
        )
    }

    if "pgd" in standard:
        comparison["pgd_accuracy_delta"] = _delta_map(
            standard, robust, "pgd", "accuracy"
        )

    if "certification" in standard:
        # Bound-based delta kept for continuity, but at eps where the standard
        # model's lower bound is floored at 0 this is NOT a delta -- it is the
        # robust value with a censored baseline. Prefer the point estimate.
        # mean_net_*: sigma-confound decomposition, i.e. how much of the
        # P_safe delta is the mean network moving vs the posterior contracting.
        for out_key, field in (
            ("mean_p_safe_lower_delta", "mean_p_safe_lower"),
            ("mean_p_safe_point_delta", "mean_p_safe_point"),
            ("mean_net_frac_safe_delta", "mean_net_frac_safe"),
        ):
            comparison[out_key] = _delta_map(
                standard, robust, "certification", field
            )

    return comparison
