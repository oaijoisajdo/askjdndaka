"""
Clean and PGD evaluation of the posterior-predictive rule.

``pgd_examples`` returns adversarial inputs and ``evaluate`` scores them.
They are separate so that a single attack pass over the eval split can feed
both the aggregate report and the per-input certification-subset analysis.
"""

from __future__ import annotations

import jax

from attacks import pgd_attack, mc_dropout_logits_fn, bbb_logits_fn
from experiments.config import CONFIG
from experiments.predictive_utils import mc_probs_batched, clean_report


def make_logits_fn(model):
    if model.model_type in ("mc_dropout", "deterministic"):
        return mc_dropout_logits_fn(model)
    if model.model_type in ("bbb", "vogn"):
        return bbb_logits_fn(model)
    raise ValueError(f"Unknown model_type: {model.model_type}")


def pgd_examples(model, x, y, *, eps: float, attack_seed: int):
    """
    EOT-PGD against the deployed posterior-predictive rule.

    The attack is per-example: row i of the output depends only on row i of
    the input, so slicing the result is equivalent to attacking the slice.
    """
    a = CONFIG.attack
    return pgd_attack(
        make_logits_fn(model),
        x,
        y,
        eps=eps,
        key=jax.random.PRNGKey(attack_seed),
        stochastic=True,
        mc_samples=CONFIG.mc_samples,
        mode=a.mode,
        targeted=False,
        steps=a.steps,
        random_start=a.random_start,
    )


def evaluate(model, x, y, *, seed: int) -> dict:
    """Aggregate report for the ensemble predictor on the given inputs."""
    probs = mc_probs_batched(
        model, x,
        n_samples=CONFIG.mc_samples,
        seed=seed,
        model_type=model.model_type,
    )
    return clean_report(probs, y)
