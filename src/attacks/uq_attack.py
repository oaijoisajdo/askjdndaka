from __future__ import annotations

from typing import Callable, Literal

import jax
import jax.numpy as jnp

from .attack_utils import LogitsFn


Array = jax.Array


UQObjective = Literal[
    # Pure uncertainty manipulation
    "max_entropy",
    "min_entropy",
    "max_mi",
    "min_mi",
    "max_variance",
    "min_variance",

    # Joint accuracy + uncertainty attacks
    "ce",
    "ce_minus_entropy",
    "ce_plus_entropy",
    "ce_minus_mi",
    "ce_plus_mi",
    "ce_minus_variance",
    "ce_plus_variance",

    # Targeted variants
    "targeted_ce",
    "targeted_ce_minus_entropy",
    "targeted_ce_minus_mi",
    "targeted_ce_minus_variance",
]


def _safe_log(x: Array, eps: float = 1e-8) -> Array:
    return jnp.log(jnp.clip(x, eps, 1.0))


def _entropy_from_probs(probs: Array) -> Array:
    """
    probs: [..., n_classes]
    returns: [...]
    """
    return -jnp.sum(probs * _safe_log(probs), axis=-1)


def _softmax_logits(logits: Array) -> Array:
    return jax.nn.softmax(logits, axis=-1)


def _one_hot(y: Array, n_classes: int) -> Array:
    return jax.nn.one_hot(y.astype(jnp.int32), n_classes)


def _cross_entropy_from_probs(probs: Array, y: Array) -> Array:
    """
    probs: [batch, n_classes]
    y:     [batch]
    returns scalar mean CE
    """
    n_classes = probs.shape[-1]
    y_oh = _one_hot(y, n_classes)
    ce = -jnp.sum(y_oh * _safe_log(probs), axis=-1)
    return jnp.mean(ce)


def _call_logits_fn(
    logits_fn: LogitsFn,
    x: Array,
    *,
    key: Array | None,
    stochastic: bool,
) -> Array:
    """
    Small compatibility wrapper.

    Preferred signature:
        logits_fn(x, key=key, stochastic=stochastic)

    Also supports:
        logits_fn(x, key=key)
        logits_fn(x)
    """
    try:
        return logits_fn(x, key=key, stochastic=stochastic)
    except TypeError:
        try:
            return logits_fn(x, key=key)
        except TypeError:
            return logits_fn(x)


def _mc_logits(
    logits_fn: LogitsFn,
    x: Array,
    *,
    key: Array | None,
    stochastic: bool,
    mc_samples: int,
) -> Array:
    """
    Returns logits with shape:
        [mc_samples, batch, n_classes]

    For deterministic attacks, this still returns a leading sample axis.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    mc_samples = max(int(mc_samples), 1)
    keys = jax.random.split(key, mc_samples)

    def one_forward(k: Array) -> Array:
        return _call_logits_fn(
            logits_fn,
            x,
            key=k,
            stochastic=stochastic,
        )

    if stochastic and mc_samples > 1:
        return jax.vmap(one_forward)(keys)

    logits = one_forward(keys[0])
    return logits[None, ...]


def uq_metrics_from_logits(logits_samples: Array) -> dict[str, Array]:
    """
    logits_samples: [mc_samples, batch, n_classes]

    Returns per-example uncertainty metrics:
        mean_probs:           [batch, n_classes]
        predictive_entropy:   [batch]
        expected_entropy:     [batch]
        mutual_information:   [batch]
        predictive_variance:  [batch]
    """
    probs_samples = _softmax_logits(logits_samples)

    mean_probs = jnp.mean(probs_samples, axis=0)

    predictive_entropy = _entropy_from_probs(mean_probs)
    expected_entropy = jnp.mean(_entropy_from_probs(probs_samples), axis=0)

    mutual_information = predictive_entropy - expected_entropy

    predictive_variance = jnp.sum(
        jnp.var(probs_samples, axis=0),
        axis=-1,
    )

    return {
        "mean_probs": mean_probs,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "predictive_variance": predictive_variance,
    }


def uq_attack_loss(
    logits_fn: LogitsFn,
    x: Array,
    y: Array,
    *,
    key: Array | None,
    stochastic: bool,
    mc_samples: int,
    objective: UQObjective,
    uncertainty_weight: float = 1.0,
) -> Array:
    """
    Scalar objective to MAXIMIZE.

    Important sign convention:
        - The attack always performs gradient ASCENT on this loss.
        - "min_entropy" means maximize negative entropy.
        - "ce_minus_entropy" means make prediction wrong while suppressing entropy.
    """
    logits_samples = _mc_logits(
        logits_fn,
        x,
        key=key,
        stochastic=stochastic,
        mc_samples=mc_samples,
    )

    metrics = uq_metrics_from_logits(logits_samples)

    mean_probs = metrics["mean_probs"]
    entropy = jnp.mean(metrics["predictive_entropy"])
    mi = jnp.mean(metrics["mutual_information"])
    variance = jnp.mean(metrics["predictive_variance"])

    ce = _cross_entropy_from_probs(mean_probs, y)

    if objective == "ce":
        return ce

    if objective == "targeted_ce":
        return -ce

    if objective == "max_entropy":
        return entropy

    if objective == "min_entropy":
        return -entropy

    if objective == "max_mi":
        return mi

    if objective == "min_mi":
        return -mi

    if objective == "max_variance":
        return variance

    if objective == "min_variance":
        return -variance

    if objective == "ce_minus_entropy":
        return ce - uncertainty_weight * entropy

    if objective == "ce_plus_entropy":
        return ce + uncertainty_weight * entropy

    if objective == "ce_minus_mi":
        return ce - uncertainty_weight * mi

    if objective == "ce_plus_mi":
        return ce + uncertainty_weight * mi

    if objective == "ce_minus_variance":
        return ce - uncertainty_weight * variance

    if objective == "ce_plus_variance":
        return ce + uncertainty_weight * variance

    if objective == "targeted_ce_minus_entropy":
        return -ce - uncertainty_weight * entropy

    if objective == "targeted_ce_minus_mi":
        return -ce - uncertainty_weight * mi

    if objective == "targeted_ce_minus_variance":
        return -ce - uncertainty_weight * variance

    raise ValueError(f"Unknown UQ objective: {objective}")


def uq_input_gradient(
    logits_fn: LogitsFn,
    x: Array,
    y: Array,
    *,
    key: Array | None,
    stochastic: bool,
    mc_samples: int,
    objective: UQObjective,
    uncertainty_weight: float,
) -> Array:
    def loss_wrt_x(x_in: Array) -> Array:
        return uq_attack_loss(
            logits_fn,
            x_in,
            y,
            key=key,
            stochastic=stochastic,
            mc_samples=mc_samples,
            objective=objective,
            uncertainty_weight=uncertainty_weight,
        )

    return jax.grad(loss_wrt_x)(x)


def uq_fgsm_attack(
    logits_fn: LogitsFn,
    x: Array,
    y: Array,
    eps: float,
    *,
    key: Array | None = None,
    stochastic: bool = False,
    mc_samples: int = 1,
    mode: Literal["decision", "probabilistic"] = "probabilistic",
    targeted: bool = False,
    objective: UQObjective | None = None,
    uncertainty_weight: float = 1.0,
    clip_min: float | Array = 0.0,
    clip_max: float | Array = 1.0,
) -> Array:
    """
    Drop-in FGSM-style UQ attack.

    Suggested objectives:
        objective="ce_minus_entropy"
            wrong + confident

        objective="ce_minus_mi"
            wrong + low epistemic uncertainty

        objective="max_entropy"
            force high predictive uncertainty

        objective="min_entropy"
            force low predictive uncertainty

        objective="max_mi"
            force posterior/dropout disagreement

        objective="min_mi"
            suppress posterior/dropout disagreement

    The `mode` argument is accepted for API compatibility with your old FGSM,
    but this function always uses probabilistic mean predictions internally.
    """
    del mode

    x = jnp.asarray(x)
    y = jnp.asarray(y)

    was_single = x.ndim == 1
    if was_single:
        x = x[None, :]

    if y.ndim == 0:
        y = y[None]

    if objective is None:
        objective = "targeted_ce" if targeted else "ce"

    grad = uq_input_gradient(
        logits_fn,
        x,
        y,
        key=key,
        stochastic=stochastic,
        mc_samples=mc_samples,
        objective=objective,
        uncertainty_weight=uncertainty_weight,
    )

    x_adv = x + eps * jnp.sign(grad)

    x_adv = jnp.clip(x_adv, x - eps, x + eps)
    x_adv = jnp.clip(x_adv, clip_min, clip_max)

    return x_adv[0] if was_single else x_adv


def uq_pgd_attack(
    logits_fn: LogitsFn,
    x: Array,
    y: Array,
    eps: float,
    *,
    alpha: float | None = None,
    steps: int = 10,
    key: Array | None = None,
    stochastic: bool = False,
    mc_samples: int = 1,
    mode: Literal["decision", "probabilistic"] = "probabilistic",
    targeted: bool = False,
    objective: UQObjective | None = None,
    uncertainty_weight: float = 1.0,
    random_start: bool = True,
    clip_min: float | Array = 0.0,
    clip_max: float | Array = 1.0,
) -> Array:
    """
    Drop-in PGD-style UQ attack.

    This performs projected gradient ASCENT on the chosen UQ objective.

    Example:
        uq_pgd_attack(
            logits_fn,
            x,
            y,
            eps=0.1,
            alpha=0.01,
            steps=40,
            stochastic=True,
            mc_samples=10,
            objective="ce_minus_entropy",
            uncertainty_weight=0.5,
        )
    """
    del mode

    x = jnp.asarray(x)
    y = jnp.asarray(y)

    was_single = x.ndim == 1
    if was_single:
        x = x[None, :]

    if y.ndim == 0:
        y = y[None]

    if key is None:
        key = jax.random.PRNGKey(0)

    if objective is None:
        objective = "targeted_ce" if targeted else "ce"

    if alpha is None:
        alpha = eps / max(steps // 2, 1)

    if random_start:
        key, init_key = jax.random.split(key)
        delta = jax.random.uniform(
            init_key,
            shape=x.shape,
            minval=-eps,
            maxval=eps,
        )
        x_adv = x + delta
        x_adv = jnp.clip(x_adv, clip_min, clip_max)
    else:
        x_adv = x

    for _ in range(steps):
        key, step_key = jax.random.split(key)

        grad = uq_input_gradient(
            logits_fn,
            x_adv,
            y,
            key=step_key,
            stochastic=stochastic,
            mc_samples=mc_samples,
            objective=objective,
            uncertainty_weight=uncertainty_weight,
        )

        x_adv = x_adv + alpha * jnp.sign(grad)

        x_adv = jnp.clip(x_adv, x - eps, x + eps)
        x_adv = jnp.clip(x_adv, clip_min, clip_max)

    return x_adv[0] if was_single else x_adv