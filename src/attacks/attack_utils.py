from __future__ import annotations

from typing import Callable, Literal

import jax
import jax.numpy as jnp
import flax.nnx as nnx

Array = jax.Array

def cross_entropy_from_logits(logits: Array, labels: Array) -> Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


def cross_entropy_from_probs(probs: Array, labels: Array, eps: float = 1e-12) -> Array:
    probs_y = probs[jnp.arange(labels.shape[0]), labels]
    return -jnp.mean(jnp.log(jnp.clip(probs_y, eps, 1.0)))



LogitsFn = Callable[[Array, Array | None, bool], Array]
# signature:
# logits_fn(x, key, stochastic) -> logits


def mc_dropout_logits_fn(model) -> LogitsFn:
    """
    Adapter for MC_BNN.

    stochastic=False:
        dropout disabled.

    stochastic=True:
        dropout enabled using rngs.dropout.
    """

    def logits_fn(x: Array, key: Array | None, stochastic: bool) -> Array:
        if stochastic:
            if key is None:
                raise ValueError("MC-dropout stochastic forward pass requires a key.")

            return model(
                x,
                disable_dropout=False,
                return_logits=True,
                rngs=nnx.Rngs(dropout=key),
            )

        return model(
            x,
            disable_dropout=True,
            return_logits=True,
        )

    return logits_fn


def bbb_logits_fn(model) -> LogitsFn:
    """
    Adapter for VI_BNN / BBB.

    stochastic=False:
        posterior mean prediction, sample=False.

    stochastic=True:
        posterior sample using rngs.bayes.
    """

    def logits_fn(x: Array, key: Array | None, stochastic: bool) -> Array:
        if stochastic:
            if key is None:
                raise ValueError("BBB stochastic forward pass requires a key.")

            return model(
                x,
                sample=True,
                return_logits=True,
                rngs=nnx.Rngs(bayes=key),
            )

        return model(
            x,
            sample=False,
            return_logits=True,
        )

    return logits_fn

def expected_input_gradient(
    logits_fn: LogitsFn,
    x: Array,
    y: Array,
    *,
    key: Array | None = None,
    stochastic: bool = False,
    mc_samples: int = 1,
    mode: Literal["decision", "probabilistic"] = "decision",
    targeted: bool = False,
) -> Array:
    if stochastic and key is None:
        raise ValueError("Provide key when stochastic=True.")

    if mc_samples < 1:
        raise ValueError("mc_samples must be >= 1.")

    def objective(x_in: Array) -> Array:
        if not stochastic:
            logits = logits_fn(x_in, None, False)
            loss = cross_entropy_from_logits(logits, y)

        else:
            keys = jax.random.split(key, mc_samples)

            def one_sample(k):
                return logits_fn(x_in, k, True)

            logits_samples = jax.vmap(one_sample)(keys)

            if mode == "decision":
                probs = jax.nn.softmax(logits_samples, axis=-1).mean(axis=0)
                loss = cross_entropy_from_probs(probs, y)

            elif mode == "probabilistic":
                log_probs = jax.nn.log_softmax(logits_samples, axis=-1)
                losses = -log_probs[:, jnp.arange(y.shape[0]), y]
                loss = losses.mean()

            else:
                raise ValueError(f"Unknown mode: {mode}")

        return -loss if targeted else loss

    return jax.grad(objective)(x)


def resolve_name(model) -> str:
    cfg = model.get_config()

    parts = [cfg["model_type"]]

    if "prior" in cfg:
        parts.append(f"{cfg['prior']['name']}_prior")

    if "p_drop" in cfg and cfg["p_drop"] > 0.0:
        parts.append(f"drop_{cfg['p_drop']}")

    if "width" in cfg:
        parts.append(f"w{cfg['width']}")

    if "depth" in cfg:
        parts.append(f"d{cfg['depth']}")

    return "_".join(parts)