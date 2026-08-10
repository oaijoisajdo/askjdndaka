"""
Family-polymorphic summaries of a posterior object.

Dispatch is on the *duck type* of the posterior, not on ``model.model_type``:
``certification.Posterior`` yields a Gaussian posterior for BBB/VOGN and a
``DropoutPosterior`` for MC-dropout/deterministic, and only the attributes
distinguish them. Keeping the three-way branch in exactly one place
(``posterior_kind``) means adding a family touches one function.
"""

from __future__ import annotations

import math
from typing import Literal

import jax.numpy as jnp
import numpy as np

PosteriorKind = Literal["gaussian", "bernoulli_dropout", "unknown"]


def posterior_kind(posterior) -> PosteriorKind:
    if hasattr(posterior, "mu_w") and hasattr(posterior, "sigma_w"):
        return "gaussian"
    if hasattr(posterior, "W") and hasattr(posterior, "b"):
        return "bernoulli_dropout"
    return "unknown"


def _spread_row(index, sigma_w, *, sigma_b_mean, relative_spread, **extra):
    """One layer's row of the spread table, identical schema for every family."""
    sw = np.asarray(sigma_w)
    return {
        "layer": int(index),
        "sigma_w_mean": float(sw.mean()),
        "sigma_w_median": float(np.median(sw)),
        "sigma_w_max": float(sw.max()),
        "sigma_b_mean": float(sigma_b_mean),
        # Scale-free: sigma relative to the weight magnitude it sits on.
        # Comparable across families and across layers.
        "relative_spread_mean": float(relative_spread),
        **extra,
    }


def posterior_sigma_stats(posterior) -> dict:
    """
    Per-layer posterior spread summaries -- Phase E data, costs nothing.

      * Gaussian (BBB / VOGN): sigma is explicit.
      * Bernoulli (MC-dropout): there is no sigma, but inverted dropout
        induces a well-defined weight-space spread. Row i of W[k+1] is
        multiplied by m_i / (1 - p) with m_i ~ Bernoulli(1 - p), so that
        entry has mean w and standard deviation |w| * sqrt(p / (1 - p)).
        Reporting that puts dropout on the same axis as the Gaussian
        families, which is what the sigma-confound comparison needs.

    The ``kind`` tag means downstream analysis never has to guess which
    convention a payload used.
    """
    kind = posterior_kind(posterior)

    if kind == "gaussian":
        layers = [
            _spread_row(
                i,
                posterior.sigma_w[i],
                sigma_b_mean=np.asarray(posterior.sigma_b[i]).mean(),
                relative_spread=(
                    np.asarray(posterior.sigma_w[i])
                    / (np.abs(np.asarray(posterior.mu_w[i])) + 1e-12)
                ).mean(),
            )
            for i in range(len(posterior.mu_w))
        ]
        return {"kind": kind, "layers": layers}

    if kind == "bernoulli_dropout":
        p = float(posterior.p)
        # sd of (m / (1 - p)) for m ~ Bernoulli(1 - p)
        mask_cv = math.sqrt(p / (1.0 - p)) if p > 0.0 else 0.0
        layers = []
        for k in range(len(posterior.W)):
            w = np.asarray(posterior.W[k])
            # Mask k acts on the rows of W[k + 1]; W[0] is deterministic.
            stochastic = k > 0
            layers.append(
                _spread_row(
                    k,
                    np.abs(w) * mask_cv if stochastic else np.zeros_like(w),
                    sigma_b_mean=0.0,          # biases are not dropped
                    relative_spread=mask_cv if stochastic else 0.0,
                    stochastic=bool(stochastic),
                )
            )
        return {
            "kind": kind,
            "p_drop": p,
            "mask_cv": mask_cv,
            "layers": layers,
        }

    return {"kind": "unknown", "layers": None}


def mean_network_layers(posterior) -> list[tuple] | None:
    """
    The zero-spread limit of the posterior: one deterministic network.

      * Gaussian (BBB / VOGN): the posterior mean, (mu_w, mu_b).
      * Bernoulli (MC-dropout): dropout disabled. Under inverted dropout the
        mask contributes m / (1 - p) with mean exactly 1, so the mean network
        is the raw (W, b) with NO 1/(1 - p) rescaling -- i.e. the ordinary
        test-time forward pass. Do not build this from ``layers_for_masks``
        with an all-ones mask; that would scale every row by 1/(1 - p).

    Returns None when the posterior exposes neither convention, in which case
    the sigma-confound control is skipped rather than crashing the run.
    """
    kind = posterior_kind(posterior)

    if kind == "gaussian":
        pairs = zip(posterior.mu_w, posterior.mu_b)
    elif kind == "bernoulli_dropout":
        pairs = zip(posterior.W, posterior.b)
    else:
        return None

    return [(jnp.asarray(w), jnp.asarray(b)) for w, b in pairs]
