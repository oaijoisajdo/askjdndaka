from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal 
from collections.abc import Sequence

import jax
import jax.numpy as jnp

Array = jax.Array
InputBox = tuple[Array, Array]
Weights = Sequence[tuple[Array, Array]]
ReluLower = Literal["zero", "adaptive"]


# ---------------------------------------------------------------------------
# Deterministic sampled-network propagation
# ---------------------------------------------------------------------------

@jax.jit
def _forward_layers_kernel(layers: Weights, x: Array) -> Array:
    h = x
    n_layers = len(layers)
    for i, (w, b) in enumerate(layers):
        h = h @ w + b
        if i < n_layers - 1:
            h = jax.nn.relu(h)
    return h


def forward_layers(layers: Weights, x: Array) -> Array:
    """Exact forward pass through fixed sampled weights.

    The compiled kernel is shared by nominal-point checks and PGD.  In
    particular, point evaluation never constructs a zero-width ``WeightBox``.
    """
    if len(layers) == 0:
        raise ValueError("layers is empty.")
    return _forward_layers_kernel(layers, jnp.asarray(x))


@jax.jit
def _propagate_deterministic_ibp_kernel(
    layers: Weights,
    x_l: Array,
    x_u: Array,
) -> tuple[Array, Array]:
    h_l, h_u = x_l, x_u
    n_layers = len(layers)

    for i, (w, b) in enumerate(layers):
        w_pos = jnp.maximum(w, 0.0)
        w_neg = jnp.minimum(w, 0.0)
        next_l = h_l @ w_pos + h_u @ w_neg + b
        next_u = h_u @ w_pos + h_l @ w_neg + b
        h_l, h_u = next_l, next_u

        if i < n_layers - 1:
            h_l = jax.nn.relu(h_l)
            h_u = jax.nn.relu(h_u)

    return h_l, h_u


def propagate_deterministic_IBP(
    layers: Weights,
    x_l: Array,
    x_u: Array,
) -> tuple[Array, Array]:
    """IBP over inputs for one fixed posterior weight sample."""
    if len(layers) == 0:
        raise ValueError("layers is empty.")

    input_l = jnp.asarray(x_l).reshape(-1)
    input_u = jnp.asarray(x_u).reshape(-1)
    if input_l.shape != input_u.shape:
        raise ValueError(
            f"Input bounds must have equal shapes; got {input_l.shape} and "
            f"{input_u.shape}."
        )
    if bool(jnp.any(input_l > input_u)):
        raise ValueError("Every input lower bound must be <= its upper bound.")

    return _propagate_deterministic_ibp_kernel(layers, input_l, input_u)


def _min_input_affine(
    coeff: Array,
    const: Array,
    x_l: Array,
    x_u: Array,
) -> Array:
    """Minimize row-wise affine forms over an input box."""
    x_mid = (x_l + x_u) / 2.0
    x_rad = (x_u - x_l) / 2.0
    return coeff @ x_mid + const - jnp.abs(coeff) @ x_rad


def _max_input_affine(
    coeff: Array,
    const: Array,
    x_l: Array,
    x_u: Array,
) -> Array:
    """Maximize row-wise affine forms over an input box."""
    x_mid = (x_l + x_u) / 2.0
    x_rad = (x_u - x_l) / 2.0
    return coeff @ x_mid + const + jnp.abs(coeff) @ x_rad


@partial(
    jax.jit,
    static_argnames=("relu_lower", "tighten_with_ibp"),
)
def _propagate_deterministic_lbp_kernel(
    layers: Weights,
    x_l: Array,
    x_u: Array,
    *,
    relu_lower: ReluLower,
    tighten_with_ibp: bool,
) -> tuple[Array, Array]:
    """
    Forward linear relaxation over inputs only.

    Each row of ``a_l``/``a_u`` is an affine lower/upper bound in the original
    input variables.  Fixed weights are contracted with matrix products; no
    weight-variable coefficient blocks are allocated.
    """
    input_dim = x_l.shape[0]
    dtype = jnp.result_type(x_l, x_u, layers[0][0])

    a_l = jnp.eye(input_dim, dtype=dtype)
    a_u = a_l
    c_l = jnp.zeros(input_dim, dtype=dtype)
    c_u = c_l

    n_layers = len(layers)
    for i, (w, b) in enumerate(layers):
        w_pos = jnp.maximum(w, 0.0)
        w_neg = jnp.minimum(w, 0.0)

        next_a_l = w_pos.T @ a_l + w_neg.T @ a_u
        next_a_u = w_pos.T @ a_u + w_neg.T @ a_l
        next_c_l = w_pos.T @ c_l + w_neg.T @ c_u + b
        next_c_u = w_pos.T @ c_u + w_neg.T @ c_l + b

        val_l = _min_input_affine(next_a_l, next_c_l, x_l, x_u)
        val_u = _max_input_affine(next_a_u, next_c_u, x_l, x_u)

        if i == n_layers - 1:
            y_l, y_u = val_l, val_u
            break

        inactive = val_u <= 0.0
        active = val_l >= 0.0
        crossing = ~(inactive | active)
        denom = jnp.where(crossing, val_u - val_l, 1.0)

        alpha_u = jnp.where(
            active,
            1.0,
            jnp.where(inactive, 0.0, val_u / denom),
        )
        beta_u = jnp.where(crossing, -alpha_u * val_l, 0.0)

        if relu_lower == "zero":
            alpha_l = jnp.where(active, 1.0, 0.0)
        else:
            # On an unstable ReLU both zero and the identity are valid lower
            # lines.  This is the standard adaptive DeepPoly heuristic.
            alpha_l = jnp.where(
                active,
                1.0,
                jnp.where(inactive, 0.0, (val_u + val_l) >= 0.0),
            ).astype(dtype)

        a_l = next_a_l * alpha_l[:, None]
        a_u = next_a_u * alpha_u[:, None]
        c_l = next_c_l * alpha_l
        c_u = next_c_u * alpha_u + beta_u

    if tighten_with_ibp:
        ibp_l, ibp_u = _propagate_deterministic_ibp_kernel(
            layers,
            x_l,
            x_u,
        )
        y_l = jnp.maximum(y_l, ibp_l)
        y_u = jnp.minimum(y_u, ibp_u)

    return y_l, y_u


def propagate_deterministic_LBP(
    layers: Weights,
    x_l: Array,
    x_u: Array,
    *,
    relu_lower: ReluLower = "adaptive",
    tighten_with_ibp: bool = False,
) -> tuple[Array, Array]:
    """
    Input-only LBP for one fixed posterior weight sample.

    ``tighten_with_ibp`` is deliberately opt-in.  The statistical property
    builders expose a separate ``ibp_first`` option that can avoid this LBP
    call entirely when cheap IBP already decides the sample.
    """
    if len(layers) == 0:
        raise ValueError("layers is empty.")
    if relu_lower not in ("zero", "adaptive"):
        raise ValueError(f"Unknown relu_lower={relu_lower!r}.")

    input_l = jnp.asarray(x_l).reshape(-1)
    input_u = jnp.asarray(x_u).reshape(-1)
    if input_l.shape != input_u.shape:
        raise ValueError(
            f"Input bounds must have equal shapes; got {input_l.shape} and "
            f"{input_u.shape}."
        )
    if bool(jnp.any(input_l > input_u)):
        raise ValueError("Every input lower bound must be <= its upper bound.")
    if layers[0][0].shape[0] != input_l.shape[0]:
        raise ValueError(
            f"Input box has dimension {input_l.shape[0]}, but first W has "
            f"{layers[0][0].shape[0]} rows."
        )

    return _propagate_deterministic_lbp_kernel(
        layers,
        input_l,
        input_u,
        relu_lower=relu_lower,
        tighten_with_ibp=tighten_with_ibp,
    )


# ---------------------------------------------------------------------------
# Input and weight interval propagation (Wicker route)
# ---------------------------------------------------------------------------

def propagate_IBP(box, x_L, x_U):
    h_L, h_U = x_L, x_U
    n_layers = len(box.layers)
    for i, (w_l, w_u, b_l, b_u) in enumerate(box.layers):
        h_L, h_U = propagate_interval(w_l, w_u, b_l, b_u, h_L, h_U)
        if i < n_layers - 1:
            h_L, h_U = jax.nn.relu(h_L), jax.nn.relu(h_U)
    return h_L, h_U

def propagate_interval(w_l, w_u, b_l, b_u, x_L, x_U):
    x_mu= (x_U + x_L) / 2
    x_r = (x_U - x_L) / 2
    w_mu = (w_u + w_l) / 2
    w_r = (w_u - w_l) / 2

    h_mu = x_mu @ w_mu
    x_rad = x_r @ jnp.abs(w_mu)
    w_rad = jnp.abs(x_mu) @ w_r
    quad = jnp.abs(x_r) @ w_r # w_r >= 0 by construction

    h_l = h_mu - x_rad - w_rad - quad + b_l
    h_u = h_mu + x_rad + w_rad + quad + b_u
    return h_l, h_u


### Lower Bound Propgation over weight intervals

@dataclass
class _AffineBounds:
    """Affine lower/upper bounds over input variables and prior layer weights."""

    # Each has shape (n_neurons, n_input_vars)
    x_l: Array
    x_u: Array

    # Each block has shape (n_neurons, n_weight_vars_for_that_layer)
    w_l: list[Array]
    w_u: list[Array]

    # Each has shape (n_neurons,)
    c_l: Array
    c_u: Array

    # Concrete scalar interval bounds for the represented quantity.
    val_l: Array
    val_u: Array


def _min_affine(
    coeff_x: Array,
    coeff_ws: list[Array],
    const: Array,
    *,
    x_l: Array,
    x_u: Array,
    w_l_flat: list[Array],
    w_u_flat: list[Array],
) -> Array:
    """Minimize one affine expression per row over the input/weight box."""
    out = const
    out = out + jnp.sum(
        jnp.where(coeff_x >= 0.0, coeff_x * x_l[None, :], coeff_x * x_u[None, :]),
        axis=-1,
    )

    for coeff, lo, hi in zip(coeff_ws, w_l_flat, w_u_flat):
        out = out + jnp.sum(
            jnp.where(coeff >= 0.0, coeff * lo[None, :], coeff * hi[None, :]),
            axis=-1,
        )

    return out


def _max_affine(
    coeff_x: Array,
    coeff_ws: list[Array],
    const: Array,
    *,
    x_l: Array,
    x_u: Array,
    w_l_flat: list[Array],
    w_u_flat: list[Array],
) -> Array:
    """Maximize one affine expression per row over the input/weight box."""
    out = const
    out = out + jnp.sum(
        jnp.where(coeff_x >= 0.0, coeff_x * x_u[None, :], coeff_x * x_l[None, :]),
        axis=-1,
    )

    for coeff, lo, hi in zip(coeff_ws, w_l_flat, w_u_flat):
        out = out + jnp.sum(
            jnp.where(coeff >= 0.0, coeff * hi[None, :], coeff * lo[None, :]),
            axis=-1,
        )

    return out


def _current_weight_block(coeff_per_input: Array, n_out: int) -> Array:
    """
    Build a coefficient block for the current layer's flattened W.

    If W has shape (n_in, n_out), flattened in row-major order, then row j of
    the returned matrix has coefficient coeff_per_input[i] on W[i, j].
    """
    coeff_per_input = jnp.asarray(coeff_per_input)
    n_in = coeff_per_input.shape[0]
    eye = jnp.eye(n_out, dtype=coeff_per_input.dtype)
    # (n_in, n_out_variable_col, n_out_expression_row)
    block = coeff_per_input[:, None, None] * eye[None, :, :]
    return jnp.transpose(block, (2, 0, 1)).reshape(n_out, n_in * n_out)


def _first_layer_affine(
    input_box: InputBox,
    layer: tuple[Array, Array, Array, Array],
) -> tuple[_AffineBounds, list[Array], list[Array]]:
    """
    Linear/McCormick bounds for zeta = x @ W + b for the first layer.

    Uses the valid McCormick planes:
      xW >= x_l W + W_l x - x_l W_l
      xW <= x_l W + W_u x - x_l W_u
    """
    x_l, x_u = input_box
    w_l, w_u, b_l, b_u = layer

    x_l = jnp.asarray(x_l).reshape(-1)
    x_u = jnp.asarray(x_u).reshape(-1)
    w_l = jnp.asarray(w_l)
    w_u = jnp.asarray(w_u)
    b_l = jnp.asarray(b_l)
    b_u = jnp.asarray(b_u)

    n_in, n_out = w_l.shape
    if x_l.shape[-1] != n_in:
        raise ValueError(
            f"Input box has dimension {x_l.shape[-1]}, but first W has {n_in} rows."
        )

    coeff_x_l = w_l.T
    coeff_x_u = w_u.T

    # Both lower and upper use the same x_l * W variable term.
    coeff_w0 = _current_weight_block(x_l, n_out)

    const_l = b_l - x_l @ w_l
    const_u = b_u - x_l @ w_u

    w_l_flat = [w_l.reshape(-1)]
    w_u_flat = [w_u.reshape(-1)]

    val_l = _min_affine(
        coeff_x_l,
        [coeff_w0],
        const_l,
        x_l=x_l,
        x_u=x_u,
        w_l_flat=w_l_flat,
        w_u_flat=w_u_flat,
    )
    val_u = _max_affine(
        coeff_x_u,
        [coeff_w0],
        const_u,
        x_l=x_l,
        x_u=x_u,
        w_l_flat=w_l_flat,
        w_u_flat=w_u_flat,
    )

    aff = _AffineBounds(
        x_l=coeff_x_l,
        x_u=coeff_x_u,
        w_l=[coeff_w0],
        w_u=[coeff_w0],
        c_l=const_l,
        c_u=const_u,
        val_l=val_l,
        val_u=val_u,
    )
    return aff, w_l_flat, w_u_flat


def _relu_affine(pre: _AffineBounds, *, relu_lower: ReluLower) -> _AffineBounds:
    """Apply ReLU linear relaxations to affine preactivation bounds."""
    l = pre.val_l
    u = pre.val_u

    inactive = u <= 0.0
    active = l >= 0.0
    crossing = ~(inactive | active)

    #denom = jnp.maximum(u - l, jnp.finfo(l.dtype).eps)
    denom = jnp.where(crossing, u - l, 1.0)

    alpha_u = jnp.where(active, 1.0, jnp.where(inactive, 0.0, u / denom))
    beta_u = jnp.where(crossing, -alpha_u * l, 0.0)

    if relu_lower == "zero":
        alpha_l = jnp.where(active, 1.0, 0.0)
        beta_l = jnp.zeros_like(l)
    elif relu_lower == "adaptive":
        # Both 0 and zeta are valid lower bounds for ReLU on crossing neurons.
        # The usual DeepPoly heuristic picks zeta when it is likely tighter.
        alpha_l = jnp.where(active, 1.0, jnp.where(inactive, 0.0, (u + l) >= 0.0))
        alpha_l = alpha_l.astype(l.dtype)
        beta_l = jnp.zeros_like(l)
    else:
        raise ValueError(f"Unknown relu_lower={relu_lower!r}.")

    def scale_rows(mat: Array, scale: Array) -> Array:
        return mat * scale[:, None]

    x_l = scale_rows(pre.x_l, alpha_l)
    x_u = scale_rows(pre.x_u, alpha_u)

    w_l = [scale_rows(block, alpha_l) for block in pre.w_l]
    w_u = [scale_rows(block, alpha_u) for block in pre.w_u]

    c_l = alpha_l * pre.c_l + beta_l
    c_u = alpha_u * pre.c_u + beta_u

    return _AffineBounds(
        x_l=x_l,
        x_u=x_u,
        w_l=w_l,
        w_u=w_u,
        c_l=c_l,
        c_u=c_u,
        val_l=jnp.maximum(l, 0.0),
        val_u=jnp.maximum(u, 0.0),
    )


def _affine_after_activation(
    act: _AffineBounds,
    layer: tuple[Array, Array, Array, Array],
    *,
    input_l: Array,
    input_u: Array,
    w_l_flat: list[Array],
    w_u_flat: list[Array],
) -> tuple[_AffineBounds, list[Array], list[Array]]:
    """
    Propagate affine ReLU bounds through zeta_next = z @ W + b.

    For each product z_i W_ij, use valid McCormick planes:
      lower: zL_i W_ij + W_L_ij z_i - zL_i W_L_ij
      upper: zL_i W_ij + W_U_ij z_i - zL_i W_U_ij

    The coefficient multiplying z_i may be negative, so we select the lower or
    upper affine bound on z_i according to the sign.
    """
    w_l, w_u, b_l, b_u = layer
    w_l = jnp.asarray(w_l)
    w_u = jnp.asarray(w_u)
    b_l = jnp.asarray(b_l)
    b_u = jnp.asarray(b_u)

    n_in, n_out = w_l.shape
    if act.val_l.shape[0] != n_in:
        raise ValueError(
            f"Activation has dimension {act.val_l.shape[0]}, but next W has {n_in} rows."
        )

    # Previous variable coefficients.
    sel_x_l = jnp.where(w_l[:, :, None] >= 0.0, act.x_l[:, None, :], act.x_u[:, None, :])
    out_x_l = jnp.sum(w_l[:, :, None] * sel_x_l, axis=0)

    sel_x_u = jnp.where(w_u[:, :, None] >= 0.0, act.x_u[:, None, :], act.x_l[:, None, :])
    out_x_u = jnp.sum(w_u[:, :, None] * sel_x_u, axis=0)

    out_w_l: list[Array] = []
    out_w_u: list[Array] = []
    for block_l, block_u in zip(act.w_l, act.w_u):
        sel_l = jnp.where(w_l[:, :, None] >= 0.0, block_l[:, None, :], block_u[:, None, :])
        out_w_l.append(jnp.sum(w_l[:, :, None] * sel_l, axis=0))

        sel_u = jnp.where(w_u[:, :, None] >= 0.0, block_u[:, None, :], block_l[:, None, :])
        out_w_u.append(jnp.sum(w_u[:, :, None] * sel_u, axis=0))

    # Coefficients for the current layer's W variables.
    curr_w_block = _current_weight_block(act.val_l, n_out)
    out_w_l.append(curr_w_block)
    out_w_u.append(curr_w_block)

    c_sel_l = jnp.where(w_l >= 0.0, act.c_l[:, None], act.c_u[:, None])
    c_sel_u = jnp.where(w_u >= 0.0, act.c_u[:, None], act.c_l[:, None])

    out_c_l = b_l + jnp.sum(w_l * c_sel_l - act.val_l[:, None] * w_l, axis=0)
    out_c_u = b_u + jnp.sum(w_u * c_sel_u - act.val_l[:, None] * w_u, axis=0)

    new_w_l_flat = w_l_flat + [w_l.reshape(-1)]
    new_w_u_flat = w_u_flat + [w_u.reshape(-1)]

    val_l = _min_affine(
        out_x_l,
        out_w_l,
        out_c_l,
        x_l=input_l,
        x_u=input_u,
        w_l_flat=new_w_l_flat,
        w_u_flat=new_w_u_flat,
    )
    val_u = _max_affine(
        out_x_u,
        out_w_u,
        out_c_u,
        x_l=input_l,
        x_u=input_u,
        w_l_flat=new_w_l_flat,
        w_u_flat=new_w_u_flat,
    )

    return (
        _AffineBounds(
            x_l=out_x_l,
            x_u=out_x_u,
            w_l=out_w_l,
            w_u=out_w_u,
            c_l=out_c_l,
            c_u=out_c_u,
            val_l=val_l,
            val_u=val_u,
        ),
        new_w_l_flat,
        new_w_u_flat,
    )


def propagate_LBP(
    weight_box: object,
    x_l: Array,
    x_u: Array,
    *,
    relu_lower: ReluLower = "adaptive",
    tighten_with_ibp: bool = True,
) -> tuple[Array, Array]:
    """
    Linear bound propagation with the same call contract as ``ibp_prob``.

    Args:
        weight_box: object with .layers = [(w_l, w_u, b_l, b_u), ...].
        x_l: lower input bound for one flattened example.
        x_u: upper input bound for one flattened example.
        relu_lower: "adaptive" chooses between 0 and z lower ReLU lines for
            crossing ReLUs; "zero" matches the older, more conservative rule.
        tighten_with_ibp: intersect the final LBP interval with the ordinary IBP
            interval. This is sound because both are over-approximations and
            prevents LBP from returning a worse final box.

    Returns:
        ``(logits_l, logits_u)`` for all inputs in ``[x_l, x_u]`` and all
        weights in ``weight_box``.
    """
    if len(weight_box.layers) == 0:
        raise ValueError("weight_box.layers is empty.")

    input_l = jnp.asarray(x_l).reshape(-1)
    input_u = jnp.asarray(x_u).reshape(-1)

    if input_l.shape != input_u.shape:
        raise ValueError(
            f"Input bounds must have equal shapes; got {input_l.shape} and "
            f"{input_u.shape}."
        )
    if bool(jnp.any(input_l > input_u)):
        raise ValueError("Every input lower bound must be <= its upper bound.")

    aff, w_l_flat, w_u_flat = _first_layer_affine((input_l, input_u), weight_box.layers[0])

    for layer_idx in range(1, len(weight_box.layers)):
        act = _relu_affine(aff, relu_lower=relu_lower)
        aff, w_l_flat, w_u_flat = _affine_after_activation(
            act,
            weight_box.layers[layer_idx],
            input_l=input_l,
            input_u=input_u,
            w_l_flat=w_l_flat,
            w_u_flat=w_u_flat,
        )

    y_l, y_u = aff.val_l, aff.val_u

    if tighten_with_ibp:
        ibp_l, ibp_u = propagate_IBP(weight_box,input_l, input_u)
        y_l = jnp.maximum(y_l, ibp_l)
        y_u = jnp.minimum(y_u, ibp_u)

    return y_l, y_u