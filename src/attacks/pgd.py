from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from .attack_utils import expected_input_gradient, LogitsFn

Array = jax.Array


def pgd_attack(
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
    steps: int = 20,
    step_size: float | None = None,
    random_start: bool = True,
) -> Array:
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    was_single = x.ndim == 1
    if was_single:
        x = x[None, :]

    if y.ndim == 0:
        y = y[None]

    if step_size is None:
        step_size = eps / max(steps // 4, 1)

    if key is not None:
        init_key, attack_key = jax.random.split(key)
    else:
        init_key, attack_key = None, None

    x_orig = x

    if random_start:
        if init_key is None:
            raise ValueError("Provide key when random_start=True.")

        delta = jax.random.uniform(
            init_key,
            shape=x.shape,
            minval=-eps,
            maxval=eps,
        )
        x_adv = jnp.clip(x_orig + delta, clip_min, clip_max)
    else:
        x_adv = x_orig

    for _ in range(steps):
        if stochastic:
            attack_key, step_key = jax.random.split(attack_key)
        else:
            step_key = None

        grad = expected_input_gradient(
            logits_fn,
            x_adv,
            y,
            key=step_key,
            stochastic=stochastic,
            mc_samples=mc_samples,
            mode=mode,
            targeted=targeted,
        )

        direction = -1.0 if targeted else 1.0
        x_adv = x_adv + direction * step_size * jnp.sign(grad)

        x_adv = jnp.clip(x_adv, x_orig - eps, x_orig + eps)
        x_adv = jnp.clip(x_adv, clip_min, clip_max)

    return x_adv[0] if was_single else x_adv