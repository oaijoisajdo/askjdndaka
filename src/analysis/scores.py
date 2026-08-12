"""
Candidate risk scores and their clean / attacked / delta variants.

Every score is oriented so that HIGHER MEANS RISKIER. That is what makes
AUROC comparable across scores: confidence and margin are flipped, entropy
and MI are already risk-oriented.
"""

from __future__ import annotations

import numpy as np

# Field name in the JSON -> whether it needs flipping to become a risk score.
_RAW_FIELDS = {
    "predictive_entropy": False,
    "mutual_information": False,
    "expected_entropy": False,
    "confidence": True,
    "predictive_margin": True,
    # Reparameterization-invariant posterior-spread proxy. Higher = riskier
    # already (more disagreement across draws), so no flip.
    "predictive_variance": False,
}

SCORE_NAMES = (
    "predictive_entropy",
    "mutual_information",
    "expected_entropy",
    "one_minus_confidence",
    "one_minus_margin",
    "predictive_variance",
)

# The plan's main comparison. The other three are retained per-input for the
# sensitivity analysis but are not looped over by default.
DEFAULT_SCORES = ("mutual_information", "one_minus_confidence")

# Deterministic models have MI identically zero and expected entropy equal to
# predictive entropy; those columns are still emitted (as zeros / duplicates)
# so the table schema stays uniform, but MI-based AUROC will be NaN there.
DEGENERATE_FOR_DETERMINISTIC = ("mutual_information", "expected_entropy",
                                "predictive_variance")

VARIANTS = ("clean", "adv", "delta")


def score_name(field: str) -> str:
    return {"confidence": "one_minus_confidence",
            "predictive_margin": "one_minus_margin"}.get(field, field)


def extract_scores(block: dict | None, n: int) -> dict[str, np.ndarray]:
    """
    Risk-oriented score arrays from one ``per_input_uncertainty`` payload.

    Missing fields yield NaN columns rather than a KeyError, so runs produced
    before ``predictive_margin`` was added still load; the NaNs then propagate
    into ``n_valid`` instead of silently becoming zeros.
    """
    out = {}
    for field, flip in _RAW_FIELDS.items():
        if block is None or field not in block:
            values = np.full(n, np.nan)
        else:
            values = np.asarray(block[field], dtype=float)
            if flip:
                values = 1.0 - values
        out[score_name(field)] = values
    return out


def delta_scores(clean: dict, adv: dict) -> dict[str, np.ndarray]:
    """
    Delta U = U_adv - U_clean, per input.

    Meaningful only because both sides are computed from the same posterior
    draws (shared ``cert_eval_key`` in the run script) on the same input
    index, so the difference is not dominated by Monte Carlo noise.
    """
    return {name: adv[name] - clean[name] for name in SCORE_NAMES}
