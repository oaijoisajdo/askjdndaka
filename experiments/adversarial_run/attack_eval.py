"""
Clean and PGD evaluation of the posterior-predictive rule.

``pgd_examples`` and ``evaluate`` are kept separate on purpose: the previous
``evaluate_pgd(..., return_x=True)` flag changed the *return type* of the
function, which is why the adversarial examples ended up being generated in
two places with slightly different keyword sets.
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
    """EOT-PGD against the deployed posterior-predictive rule."""
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


def evaluate_pgd(model, x, y, *, eps: float, seed: int) -> dict:
    """Attack with one seed stream, evaluate with a disjoint one."""
    x_adv = pgd_examples(model, x, y, eps=eps, attack_seed=seed)
    return evaluate(model, x_adv, y, seed=seed + CONFIG.attack.eval_seed_offset)
