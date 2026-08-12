"""
Clean and PGD evaluation of the posterior-predictive rule.

``pgd_examples`` returns adversarial inputs and ``evaluate`` scores them.
They are separate so that a single attack pass over the eval split can feed
both the aggregate report and the per-input certification-subset analysis.
"""

from __future__ import annotations

import jax

from attacks import pgd_attack, uq_pgd_attack, mc_dropout_logits_fn, bbb_logits_fn
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

    Main engine is ``uq_pgd_attack`` with objective="ce": cross-entropy of
    the MC-mean softmax, mathematically identical to the previous
    ``pgd_attack(mode="decision")`` objective (gradient-of-loss-of-mean).
    ``alpha`` is pinned to the previous step size eps/(steps//4) so the
    engine switch changes neither the estimand nor the schedule.
    ``restarts=1`` is the headline configuration; CONFIG.attack.restarts > 1
    is the convergence-check configuration.

    The legacy ``pgd_attack`` path remains for mode="probabilistic"
    (mean of per-sample CE), which has no uq_attack equivalent.

    The attack is per-example: row i of the output depends only on row i of
    the input, so slicing the result is equivalent to attacking the slice.
    """
    a = CONFIG.attack
    if a.mode == "probabilistic":
        return pgd_attack(
            make_logits_fn(model), x, y, eps=eps,
            key=jax.random.PRNGKey(attack_seed),
            stochastic=True, mc_samples=CONFIG.mc_samples,
            mode="probabilistic", targeted=False,
            steps=a.steps, random_start=a.random_start,
            restarts=getattr(a, "restarts", 1),
        )
    return uq_pgd_attack(
        make_logits_fn(model), x, y, eps=eps,
        key=jax.random.PRNGKey(attack_seed),
        stochastic=True, mc_samples=CONFIG.mc_samples,
        objective="ce", targeted=False,
        alpha=eps / max(a.steps // 4, 1),      # preserve legacy step size
        steps=a.steps, random_start=a.random_start,
        restarts=getattr(a, "restarts", 1),
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
