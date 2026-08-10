from dataclasses import dataclass
import itertools
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from attacks import pgd_attack
from .posteriors import Posterior
from .bound_propagation import propagate_IBP, propagate_interval




def prob_veri_lower(posterior, x_L, x_U, y, d_y,
                    gamma, N_samples, safe_fn, propagation_fn = propagate_IBP, d = 2, seed = 90,
                    box_generation="sample", wicker_mode = False):
    """
    A rewrite of Wicker et al. function to compute a lower bound on certified robustness

    Iteratively 
        samples weights from posterior 
        checks safety of constructed box
    then return the summed probability mass of all safe boxes
    """

    wicker_gamma = gamma * 2  # HUGE QUESTIONMARK? Doubling not documented anywhere

    H = []
    print("Starting verification...")

    if box_generation == "centered":
        print("Warning: Box generation not sampled but based on posterior mean")
    else:
        print(f"    Generating {N_samples} weight boxes")

    key = jax.random.key(seed)
    candidates = posterior.make_weight_boxes(
        key=key, n_samples=N_samples, gamma=gamma, box_generation=box_generation
    )
    for box in candidates:

        logits_L, logits_U = propagation_fn(box, x_L, x_U) # propagate to get logits
        
        if safe_fn(logits_L, logits_U, y, d_y):
            H.append(box)
    
    if box_generation == "centered":
        print(f"    Mean centered Interval safe: {safe_fn(logits_L, logits_U, y, d_y)}")
    else:
        print(f"    Found {len(H)} safe intervals")

    if wicker_mode:
        p = compute_bonferroni_p_wicker(H,d, safety_gamma=wicker_gamma)
    
    else:
        p = compute_bonferroni_p(H, d)
    return p


    
def pred_safe(logits_L, logits_U, y, y_dim):
    """
    Checks if the lower bound of true class logit
    is larger than
    the upper bound of every false class logit

    logits_L[y] > max(logits_U[not_y])

    Broadcasts over any leading batch/cell dimension; callers that need a
    single bool wrap the (scalar) result in bool(...).
    """

    classes = jnp.arange(y_dim) # vector y_dim

    other_u = jnp.max( # find max upper logit of not y
        jnp.where(classes == y, -jnp.inf, logits_U), # mask y logit
        axis=-1,
    )

    return logits_L[..., y] > other_u


def pred_unsafe(logits_L, logits_U, y, y_dim):
    """
    Checks if the lower bound of any false class logit
    is larger than
    the upper bound of the true class logit

    logits_U[y] < max(logits_L[not_y])

    Broadcasts over any leading batch/cell dimension; callers that need a
    single bool wrap the (scalar) result in bool(...).
    """
    classes = jnp.arange(y_dim)
    other_l = jnp.max(
        jnp.where(classes == y, -jnp.inf, logits_L),
        axis=-1,
    )
    return other_l > logits_U[..., y]
    




def compute_bonferroni_p(boxes, d):
    # Keeps model evaluation in float32
    with jax.enable_x64(True):
        return _compute_bonferroni_p_x64(boxes, d)


def _compute_bonferroni_p_x64(boxes, d):
    if d < 1:
        raise ValueError(
            "A Bonferroni lower bound requires a positive depth."
        )
    
    n = len(boxes)
    if n == 0:
        return jnp.array(0.0, dtype=jnp.float64)

    # Odd depth breaks the lower-bound guarantee while the series is truncated
    #  At d >= n the full expansion is exact

    if d < n and d % 2 != 0:
        raise ValueError(
            "A truncated Bonferroni lower bound requires even depth."
        )
    print("     Bonferroni calculation...")

    # Explicit conversion for x64
    z_l = jnp.stack([box.z_l_flat for box in boxes]).astype(jnp.float64)
    z_u = jnp.stack([box.z_u_flat for box in boxes]).astype(jnp.float64)

    total = jnp.array(0.0, dtype=jnp.float64)

    for k in range(1, min(d, n) + 1):
        combos = jnp.asarray(list(itertools.combinations(range(n), k)),dtype=jnp.int32,) # (C, k)

        inter_l = jnp.max(z_l[combos], axis=1)  # (C, P)
        inter_u = jnp.min(z_u[combos], axis=1)

        log_probs = jax.vmap(Posterior.log_standard_normal_interval_prob)(inter_l, inter_u)

        # Stable sum
        order_sum = jnp.exp(logsumexp(log_probs))

        if k % 2 == 1:
            total += order_sum
        else:
            total -= order_sum
        
        print(f"        Iteration {k}:")
        print(f"        current box log prop range: min = {jnp.min(log_probs)}, median = {jnp.median(log_probs)}, max = {jnp.max(log_probs)}")
        print(f"        Current total = {total}")


    return max(total, 0.0)




def compute_bonferroni_p_wicker(
    boxes,
    d,
    *,
    safety_gamma,
):
    """
    Reproduce Wicker's differing safety/probability box radii.

    The supplied boxes are the safety boxes, constructed with radius
    `safety_gamma` in standardized posterior coordinates.

    Wicker's original implementation effectively uses:

        safety radius      = safety_gamma
        probability radius = 2 * safety_gamma

    The inclusion-exclusion calculation uses corrected intersections and
    numerically stable float64 probability arithmetic. It is therefore a
    reproduction of Wicker's intended geometry, not every implementation bug.
    """
    if safety_gamma < 0:
        raise ValueError("safety_gamma must be non-negative.")

    if d < 1:
        raise ValueError(
            "A Bonferroni lower bound requires a positive depth."
        )
    
    n = len(boxes)
    if n == 0:
        return jnp.array(0.0, dtype=jnp.float64)

    # Odd depth breaks the lower-bound guarantee while the series is truncated
    #  At d >= n the full expansion is exact

    if d < n and d % 2 != 0:
        raise ValueError(
            "A truncated Bonferroni lower bound requires even depth."
        )

    with jax.enable_x64(True):
        if n == 0:
            return jnp.array(0.0, dtype=jnp.float64)

        stored_l = jnp.stack(
            [box.z_l_flat for box in boxes]
        ).astype(jnp.float64)

        stored_u = jnp.stack(
            [box.z_u_flat for box in boxes]
        ).astype(jnp.float64)

        # Recover sampled standardized centers from the safety boxes.
        z_center = (stored_l + stored_u) / 2.0

        # This is the Wicker-specific mismatch.
        probability_gamma = jnp.asarray(
            2.0 * safety_gamma,
            dtype=jnp.float64,
        )

        z_l = z_center - probability_gamma
        z_u = z_center + probability_gamma

        max_order = min(d, n)
        order_log_sums = []
        order_signs = []

        for k in range(1, max_order + 1):
            combo_indices = jnp.asarray(
                list(itertools.combinations(range(n), k)),
                dtype=jnp.int32,
            )

            intersection_l = jnp.max(
                z_l[combo_indices],
                axis=1,
            )
            intersection_u = jnp.min(
                z_u[combo_indices],
                axis=1,
            )

            valid = jnp.all(
                intersection_l < intersection_u,
                axis=1,
            )

            log_probabilities = jax.vmap(
                Posterior.log_standard_normal_interval_prob
            )(intersection_l, intersection_u)

            log_probabilities = jnp.where(
                valid,
                log_probabilities,
                -jnp.inf,
            )

            # Sum every probability at this intersection order in log space.
            order_log_sums.append(
                logsumexp(log_probabilities)
            )
            order_signs.append(
                1.0 if k % 2 == 1 else -1.0
            )

        # Perform the final alternating sum as a signed log-sum-exp.
        log_abs_total, total_sign = logsumexp(
            jnp.stack(order_log_sums),
            b=jnp.asarray(order_signs, dtype=jnp.float64),
            return_sign=True,
        )

        total = total_sign * jnp.exp(log_abs_total)

        # A negative Bonferroni lower bound contains no useful information.
        return jnp.maximum(total, 0.0)
    


def prob_veri_upper(posterior, x_L, x_U, y, d_y, gamma, N_samples, safe_fn=pred_unsafe, d = 2, seed = 90):

    wicker_gamma = gamma * 2  # HUGE QUESTIONMARK? Doubling not documented anywhere

    J = []

    key = jax.random.key(seed)
    candidates = posterior.make_weight_boxes(
        key=key, n_samples=N_samples, gamma=gamma, box_generation="sample"
    )
    for box in candidates:

        logits_L, logits_U = propagate_IBP(box, x_L, x_U) # propagate to get logits
        # TODO Add pgd attack here
        if safe_fn(logits_L, logits_U, y, d_y):
            J.append(box)
    
    p = 1 - compute_bonferroni_p(J, d) # upper bound on P(S) = 1 - p(-S) 
                                       # P() = P(S) + P(-S) + p(UNK)
    return p


# ===========================================================================
# Exact-partition certification for the MC-dropout (Bernoulli) posterior
# ===========================================================================
# A partition of the mask space (DropoutPosterior.partition_states) gives
# pairwise-disjoint weight sets whose masses sum to exactly 1, so no
# Bonferroni correction is needed -- see WORKPLAN.md.


@dataclass
class DropoutCertResult:
    p_safe_lower: float
    p_safe_upper: float
    n_cells: int
    n_safe_cells: int
    n_unsafe_cells: int
    total_mass: float  # sanity: must be ~1 for a full partition

def prob_veri_dropout(
    posterior,
    x_L,
    x_U,
    y,
    d_y,
    *,
    fixed_units=None,
    n_fixed: int | list[int] = 8,
    propagation_fn=propagate_IBP,
    batched: bool = True,
    check_full_support_first: bool = False,
):
    """
    Sound lower AND upper bounds on P_safe for a DropoutPosterior via an
    exhaustive disjoint partition of the mask space.

    Args:
        posterior: a DropoutPosterior.
        fixed_units: per-hidden-layer index arrays of units to pin. If None,
            chosen by posterior.choose_fixed_units(x_L, x_U, n_fixed).
        n_fixed: budget per hidden layer when fixed_units is None. The number
            of cells is 2**(total fixed), so this is the dropout analogue of
            the Gaussian gamma/N_samples trade-off.
        propagation_fn: propagate_IBP (vmappable fast path) or propagate_LBP
            (falls back to a per-cell loop).
        batched: use the vmapped fast path when propagation_fn is
            propagate_IBP.
        check_full_support_first: try the single all-free box first; if it
            certifies, return P_safe = 1 without enumerating.
    """
    if check_full_support_first:
        full = posterior.full_support_box()
        lL, lU = propagation_fn(full, x_L, x_U)
        if bool(pred_safe(lL, lU, y, d_y)):
            return DropoutCertResult(1.0, 1.0, 1, 1, 0, 1.0)
        if bool(pred_unsafe(lL, lU, y, d_y)):
            return DropoutCertResult(0.0, 0.0, 1, 0, 1, 1.0)

    if fixed_units is None:
        fixed_units = posterior.choose_fixed_units(x_L, x_U, n_fixed)

    if batched and propagation_fn is propagate_IBP:
        lL, lU, log_probs = _dropout_partition_ibp_bounds(
            posterior, x_L, x_U, fixed_units
        )
        n_cells = int(log_probs.shape[0])
        safe = np.asarray(pred_safe(lL, lU, y, d_y))
        unsafe = np.asarray(pred_unsafe(lL, lU, y, d_y))
    else:
        boxes = posterior.partition_weight_boxes(fixed_units)
        n_cells = len(boxes)
        log_probs = np.array(
            [float(b.log_prob) for b in boxes], dtype=np.float64
        )
        safe = np.zeros(n_cells, dtype=bool)
        unsafe = np.zeros(n_cells, dtype=bool)
        for i, box in enumerate(boxes):
            lL, lU = propagation_fn(box, x_L, x_U)
            safe[i] = bool(pred_safe(lL, lU, y, d_y))
            unsafe[i] = bool(pred_unsafe(lL, lU, y, d_y))

    masses = np.exp(log_probs)
    total_mass = float(masses.sum())

    #  a full partition's masses sum to 1 exactly 

    assert abs(total_mass - 1.0) < 1e-6, f"partition mass {total_mass} != 1"

    mass_safe = float(masses[safe].sum())
    mass_unsafe = float(masses[unsafe].sum())

    return DropoutCertResult(
        p_safe_lower=mass_safe,
        p_safe_upper=1.0 - mass_unsafe,
        n_cells=n_cells,
        n_safe_cells=int(safe.sum()),
        n_unsafe_cells=int(unsafe.sum()),
        total_mass=total_mass,
    )


def _dropout_partition_ibp_bounds(posterior, x_L, x_U, fixed_units):
    """
    IBP logit bounds for every cell of a dropout partition at once.

    Invariant: layer 0 and every bias are mask-independent, so the shared
    prefix is propagated once outside the vmap over cells, not per-cell.
    Folding it into the per-cell function measured ~170x slower on 4096
    cells (120s -> 0.7s; see WORKPLAN.md #2). Returns (logits_L, logits_U)
    of shape (n_cells, d_out) and the float64 cell log-masses.
    """
    states, log_probs = posterior.partition_states(fixed_units)

    x_L = jnp.asarray(x_L).reshape(-1)
    x_U = jnp.asarray(x_U).reshape(-1)

    # Shared deterministic first layer + ReLU.
    W0, b0 = posterior.W[0], posterior.b[0]
    h_L, h_U = propagate_interval(W0, W0, b0, b0, x_L, x_U)
    h_L, h_U = jax.nn.relu(h_L), jax.nn.relu(h_U)

    # Per-cell interval weights for each masked layer.
    masked = [
        posterior.stacked_mask_layer_bounds(states[k], k)
        for k in range(posterior.n_mask_layers)
    ]
    biases = [posterior.b[k + 1] for k in range(posterior.n_mask_layers)]
    n_masked = len(masked)

    @jax.jit
    def _all_cells(h_L, h_U, masked, biases):
        def one(cell_masked):
            hl, hu = h_L, h_U
            for k in range(n_masked):
                w_l, w_u = cell_masked[k]
                b = biases[k]
                hl, hu = propagate_interval(w_l, w_u, b, b, hl, hu)
                if k < n_masked - 1:
                    hl, hu = jax.nn.relu(hl), jax.nn.relu(hu)
            return hl, hu

        return jax.vmap(one)(masked)

    logits_L, logits_U = _all_cells(h_L, h_U, masked, biases)
    return logits_L, logits_U, log_probs


