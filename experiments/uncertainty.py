"""
Per-input predictive uncertainty, computed from the posterior object.

Using the posterior here (rather than the model's own MC path) means the
uncertainty measures and the certified P_safe values in the same payload are
computed from an identical representation of the weight distribution -- which
is exactly what the Phase C alignment analysis compares.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from certification import forward_layers   # re-export this from __init__

_EPS = 1e-12


def mc_probs_from_posterior(posterior, x, *, n_samples: int, key) -> np.ndarray:
    """
    Per-sample predictive probabilities, shape (S, N, C).

    Both Gaussian and Bernoulli posteriors duck-type
    ``sample(key) -> [(w, b), ...]``, and ``forward_layers`` batches over the
    leading input axis, so this works for every family without touching model
    internals.

    For a deterministic model every draw coincides, so the mutual information
    is 0 by construction; that is the correct answer, not a bug.
    """
    xs = jnp.asarray(x)
    probs = jnp.stack([
        jax.nn.softmax(jnp.asarray(forward_layers(posterior.sample(k), xs)), -1)
        for k in jax.random.split(key, n_samples)
    ])
    return np.asarray(probs)                        # (S, N, C)


def per_input_uncertainty(probs_snc, y) -> dict[str, list]:
    """
    Entropy, mutual information, confidence and correctness per input.

    Expects probs of shape (S, N, C) exactly as produced by
    ``mc_probs_from_posterior`` -- no axis guessing.

        H[E_w p]                        total predictive uncertainty
        H[E_w p] - E_w H[p]             mutual information (epistemic)
    """
    p = np.asarray(probs_snc)
    if p.ndim != 3:
        raise ValueError(f"Expected (S, N, C) probabilities; got {p.shape}.")
    if p.shape[1] != len(y):
        raise ValueError(f"Expected {len(y)} inputs on axis 1; got {p.shape}.")

    mean_p = p.mean(axis=0)                             # (N, C)
    entropy = -(mean_p * np.log(mean_p + _EPS)).sum(-1)
    expected_entropy = -(p * np.log(p + _EPS)).sum(-1).mean(axis=0)   # (N,)

    return {
        "n_mc_samples": int(p.shape[0]),
        "predictive_entropy": entropy.tolist(),
        "mutual_information": (entropy - expected_entropy).tolist(),
        "expected_entropy": expected_entropy.tolist(),
        "confidence": mean_p.max(-1).tolist(),
        "correct": (mean_p.argmax(-1) == np.asarray(y)).astype(int).tolist(),
    }
