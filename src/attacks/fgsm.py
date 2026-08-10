from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from .attack_utils import expected_input_gradient, LogitsFn


Array = jax.Array



def fgsm_attack(
    logits_fn: LogitsFn,
    x: Array,
    y: Array,
    eps: float,
    *,
    key: Array | None = None,
    stochastic: bool = False,
    mc_samples: int = 1,
    mode: Literal["decision", "probabilistic"] = "decision",
    targeted: bool = False,
    clip_min: float | Array = 0.0,
    clip_max: float | Array = 1.0,
) -> Array:
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    was_single = x.ndim == 1
    if was_single:
        x = x[None, :]

    if y.ndim == 0:
        y = y[None]

    grad = expected_input_gradient(
        logits_fn,
        x,
        y,
        key=key,
        stochastic=stochastic,
        mc_samples=mc_samples,
        mode=mode,
        targeted=targeted,
    )

    direction = -1.0 if targeted else 1.0
    x_adv = x + direction * eps * jnp.sign(grad)

    x_adv = jnp.clip(x_adv, x - eps, x + eps)
    x_adv = jnp.clip(x_adv, clip_min, clip_max)

    return x_adv[0] if was_single else x_adv