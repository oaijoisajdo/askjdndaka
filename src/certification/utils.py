import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

Array = jax.Array
InputBox = tuple[Array, Array]


# ---------------------------------------------------------------------
# Input/output specifications
# ---------------------------------------------------------------------

def make_linf_box(
    x: Array,
    eps: float,
    *,
    clip_min: float | None = 0.0,
    clip_max: float | None = 1.0,
    flatten: bool = True,
) -> InputBox:
    x = jnp.asarray(x)
    if flatten:
        x = x.reshape(-1)

    x_l = x - eps
    x_u = x + eps

    if clip_min is not None:
        x_l = jnp.maximum(x_l, clip_min)
    if clip_max is not None:
        x_u = jnp.minimum(x_u, clip_max)

    return x_l, x_u


def true_class_polytope(
    true_label: int,
    n_classes: int,
    *,
    margin: float = 0.0,
) -> tuple[Array, Array]:
    """
    Safe set:
        S = {y : y_true - y_k >= margin for all k != true}

    Works on logits or probabilities.
    """
    rows = []
    ds = []

    for k in range(n_classes):
        if k == true_label:
            continue

        row = jnp.zeros((n_classes,))
        row = row.at[true_label].set(1.0)
        row = row.at[k].set(-1.0)

        rows.append(row)
        ds.append(-margin)

    return jnp.stack(rows, axis=0), jnp.asarray(ds)


def interval_linear_bounds(
    C: Array,
    d: Array,
    y_l: Array,
    y_u: Array,
) -> tuple[Array, Array]:
    """
    Bounds C y + d over y in [y_l, y_u].
    """
    C_pos = jnp.maximum(C, 0.0)
    C_neg = jnp.minimum(C, 0.0)

    lower = C_pos @ y_l + C_neg @ y_u + d
    upper = C_pos @ y_u + C_neg @ y_l + d

    return lower, upper


def interval_is_inside_polytope(
    y_l: Array,
    y_u: Array,
    C: Array,
    d: Array,
    *,
    tol: float = 1e-8,
) -> bool:
    """
    Certifies [y_l, y_u] subset S where S = {y : C y + d >= 0}.
    """
    lower, _ = interval_linear_bounds(C, d, y_l, y_u)
    return bool(jnp.all(lower >= -tol))


def interval_is_disjoint_from_polytope(
    y_l: Array,
    y_u: Array,
    C: Array,
    d: Array,
    *,
    tol: float = 1e-8,
) -> bool:
    """
    Sufficient unsafe check:
    [y_l, y_u] lies outside S if at least one constraint is
    definitely violated over the whole interval.
    """
    _, upper = interval_linear_bounds(C, d, y_l, y_u)
    return bool(jnp.any(upper < -tol))




# ---------------------------------------------------------------------
# Box probability / disjointness utilities
# ---------------------------------------------------------------------

def boxes_overlap_in_z(a: object, b: object) -> bool:
    """
    Checks overlap in posterior z-space.

    This matches your Posterior.sample_weight_box(), which stores
    z_l_flat and z_u_flat.
    """
    if (
        a.z_l_flat is None
        or a.z_u_flat is None
        or b.z_l_flat is None
        or b.z_u_flat is None
    ):
        raise ValueError("Need store_z_bounds=True to enforce disjoint boxes.")

    return bool(jnp.all((a.z_l_flat <= b.z_u_flat) & (b.z_l_flat <= a.z_u_flat)))


def is_disjoint_from_all(box: object, accepted: list[object]) -> bool:
    return all(not boxes_overlap_in_z(box, other) for other in accepted)


def sum_disjoint_box_probs(boxes: list[object]) -> Array:
    """
    Sum P(box) for pairwise-disjoint boxes.
    """
    if not boxes:
        return jnp.array(0.0)

    logps = jnp.stack([jnp.asarray(b.log_prob) for b in boxes])
    return jnp.exp(logsumexp(logps))


def center_layers_from_box(box: object) -> list[tuple[Array, Array]]:
    """
    Recover the sampled deterministic network at the center of a WeightBox.
    Useful for attack generation.
    """
    return [
        ((w_l + w_u) / 2.0, (b_l + b_u) / 2.0)
        for (w_l, w_u, b_l, b_u) in box.layers
    ]


def forward_with_layers(x: Array, layers: list[tuple[Array, Array]]) -> Array:
    """
    Deterministic forward pass using sampled center weights.
    Returns logits.
    """
    z = x

    for i, (w, b) in enumerate(layers):
        z = z @ w + b
        if i < len(layers) - 1:
            z = jnp.maximum(z, 0.0)

    return z

