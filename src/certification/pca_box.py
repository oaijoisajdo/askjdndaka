"""
Sound input-box construction for BNNs that operate on PCA features.

The Bayesian net sees z = (x - mean_) @ components_.T, a frozen affine map
fit once in the data loader. PCA contributes zero Bayesian weight dimensions,
which is what makes small-net certification escape the underflow/zero-safe-box
squeeze on full 784-input nets.

The threat model stays in PIXEL space: an L_inf ball of radius epsilon around
the original pixel image, clipped to valid pixel range, is pushed through the
affine PCA map. The interval hull of an affine image of a box has a closed
form (center / radius), so this is exact interval arithmetic on a constant
layer -- sound, with no adaptivity.

Soundness note: the net only ever sees z, so any pixel perturbation (including
motion along the discarded null-space directions) maps into this z-box.
Certifying "all z in the box -> safe" therefore certifies the full pixel ball.
The interval hull ignores inter-component correlation (the true reachable set
is a zonotope with only 784 generators), so it is sound but not tight; that
looseness grows with k. Clipping to [0, 1] before projecting tightens it for
free near the pixel boundary.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def pca_input_box(
    x_pixels: Array,
    epsilon: float,
    components: Array,   # sklearn PCA.components_, shape (k, n_pixels)
    mean: Array,         # sklearn PCA.mean_,        shape (n_pixels,)
    *,
    clip: tuple[float, float] = (0.0, 1.0),
) -> tuple[Array, Array]:
    """
    z-space input box for the L_inf pixel ball of radius epsilon around
    x_pixels, projected through frozen PCA.

    Args:
        x_pixels: ORIGINAL flattened pixel image (n_pixels,). Do not pass a
            PCA reconstruction; the clip must use true pixel values.
        epsilon: pixel-space L_inf radius (same units/space as robust eps).
        components: PCA.components_, orthonormal rows (no whitening).
        mean: PCA.mean_.
        clip: valid pixel range, applied before projection.

    Returns:
        (z_l, z_u): per-feature lower/upper bounds, each shape (k,), suitable
        as the (x_L, x_U) arguments to propagate_IBP / propagate_LBP.

    Equivalent to prepending a deterministic dense layer with
    W = components.T (n_pixels x k), b = -mean @ components.T and running the
    project's dense_ibp_local on the clipped pixel box.
    """
    x = jnp.asarray(x_pixels).reshape(-1)
    P = jnp.asarray(components)          # (k, n_pixels)
    mu = jnp.asarray(mean).reshape(-1)   # (n_pixels,)
    lo, hi = clip

    if P.shape[1] != x.shape[0]:
        raise ValueError(
            f"components has {P.shape[1]} pixel columns but x has "
            f"{x.shape[0]} pixels."
        )
    if mu.shape[0] != x.shape[0]:
        raise ValueError("mean and x_pixels must have the same length.")

    x_l = jnp.clip(x - epsilon, lo, hi)
    x_u = jnp.clip(x + epsilon, lo, hi)

    x_c = 0.5 * (x_l + x_u)              # clipped midpoint (tighter than x)
    x_r = 0.5 * (x_u - x_l)              # clipped half-width (0 where clamped)

    z_c = (x_c - mu) @ P.T              # exact projected center
    z_r = x_r @ jnp.abs(P).T           # affine-image radius (>= 0)

    return z_c - z_r, z_c + z_r


def pca_feature_box(
    x_pixels: Array,
    epsilon: float,
    components: Array,
    mean: Array,
) -> tuple[Array, Array]:
    """
    Alternative FEATURE-space threat model: perturb the k PCA coordinates
    directly by +/- epsilon. Self-consistent with training that perturbs z,
    but note eps here acts on z-coordinates whose std can be tens, so a small
    eps is a near-trivial perturbation and the robustness claim is
    'robust to top-k PCA-feature perturbations', not a pixel claim. Provided
    for completeness; pca_input_box (pixel threat model) is the default.
    """
    x = jnp.asarray(x_pixels).reshape(-1)
    P = jnp.asarray(components)
    mu = jnp.asarray(mean).reshape(-1)
    z_c = (x - mu) @ P.T
    return z_c - epsilon, z_c + epsilon