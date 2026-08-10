import math

import jax
import jax.numpy as jnp
import numpy as np

from .gaussian import WeightBox

Array = jax.Array

# ===========================================================================
# MC-dropout as an approximate BNN posterior (Gal & Ghahramani, 2016)
# ===========================================================================
#
# The dropout "posterior" over weights is discrete with finite support:
# dropping hidden unit i of hidden layer k zeroes row i of W_{k+1}, keeping
# it scales that row by 1/(1 - p) (inverted-dropout scaling, matching
# nnx.Dropout in training mode). Every mask configuration is an atom with
# exactly computable Bernoulli mass, so posterior integration over weight
# sets is exact -- no ndtr products and no gamma margin.
#
# The analogue of a sampled Gaussian weight box is a *mask box*: a subset of
# units is FIXED (mask pinned to kept or dropped) and the rest are FREE
# (mask unresolved). In weight space, for the layer consuming those
# activations:
#
#   kept    -> row exactly  W[i, :] / (1 - p)      (zero-width interval)
#   dropped -> row exactly  0                      (zero-width interval)
#   free    -> elementwise  [min(0, w'), max(0, w')],  w' = W[i, :] / (1 - p)
#
# The free-row interval contains both mask outcomes, so IBP/LBP over the
# resulting WeightBox soundly bounds every network in the cell. Enumerating
# all 2^{|F|} assignments of a fixed set F yields a pairwise-disjoint
# partition of the whole mask space with total mass exactly 1, so the
# Bonferroni machinery collapses to a plain sum of certified cell masses.



# Per hidden layer: int8 state per unit. 1 = kept, 0 = dropped, -1 = free.
MaskStates = list

class DropoutPosterior:
    """
    Bernoulli (MC-dropout) posterior over masked networks.

    Duck-types the parts of ``Posterior`` consumed by the certification
    pipelines:

      * ``sample(key)``          -> ``[(w0, b0), (w1, b1), ...]`` materialised
                                    deterministic network (statistical route,
                                    ``probabilistic_safety.estimate_p_bnn_certified``).
      * mask boxes are ``WeightBox`` objects consumable by ``propagate_IBP``
        and ``propagate_LBP`` unchanged (sound route). Their ``log_prob`` is
        the exact Bernoulli mass; ``z_l_flat``/``z_u_flat`` are ``None``
        because the Gaussian Bonferroni intersection logic does not apply.

    Expects a model shaped like ``MC_BNN``: ``model.layers`` (hidden
    ``nnx.Linear``), ``model.output_layer``, ``model.depth``,
    ``model.p_drop``, with dropout applied to the activations of every
    hidden layer.
    """

    def __init__(self, model, *, max_partition_cells: int = 2 ** 16):
        self.p = float(model.p_drop)
        if not (0.0 <= self.p < 1.0):
            raise ValueError(f"p_drop must be in [0, 1); got {self.p}.")

        self.scale = 1.0 / (1.0 - self.p)
        self.max_partition_cells = int(max_partition_cells)

        linear_layers = [model.layers[i] for i in range(model.depth)]
        linear_layers.append(model.output_layer)

        self.W = [self._as_array(lay.kernel) for lay in linear_layers]
        self.b = [self._as_array(lay.bias) for lay in linear_layers]

        # Mask k lives on the activations of hidden layer k and therefore
        # acts on the rows of W[k + 1]. W[0] and every bias are deterministic.
        self.n_mask_layers = len(self.W) - 1
        self.mask_dims = [self.W[k + 1].shape[0] for k in range(self.n_mask_layers)]

        self._log_keep = math.log1p(-self.p)
        self._log_drop = math.log(self.p) if self.p > 0.0 else -math.inf

    @staticmethod
    def _as_array(x):
        if hasattr(x, "value"):
            return jnp.asarray(x.value)
        return jnp.asarray(x)

    # ------------------------------------------------------------------
    # Statistical route: sample deterministic networks
    # ------------------------------------------------------------------

    def sample_masks(self, key: Array) -> list[Array]:
        """One Bernoulli keep-mask (bool) per hidden layer."""
        keys = jax.random.split(key, self.n_mask_layers)
        return [
            jax.random.bernoulli(k, 1.0 - self.p, (n,))
            for k, n in zip(keys, self.mask_dims)
        ]

    def layers_for_masks(self, masks: list[Array]) -> list[tuple[Array, Array]]:
        """
        Materialise the deterministic network equivalent to one mask draw.

        Matches ``MC_BNN.__call__`` with active dropout: activation i of
        hidden layer k is multiplied by mask_i / (1 - p), which is absorbed
        into the rows of W[k + 1].
        """
        if len(masks) != self.n_mask_layers:
            raise ValueError(
                f"Expected {self.n_mask_layers} masks, got {len(masks)}."
            )

        out = [(self.W[0], self.b[0])]
        for k, m in enumerate(masks):
            m = jnp.asarray(m)
            if m.shape != (self.mask_dims[k],):
                raise ValueError(
                    f"Mask {k} has shape {m.shape}; expected "
                    f"({self.mask_dims[k]},)."
                )
            row_scale = m.astype(self.W[k + 1].dtype) * self.scale
            out.append((self.W[k + 1] * row_scale[:, None], self.b[k + 1]))
        return out

    def sample(self, key: Array) -> list[tuple[Array, Array]]:
        """Sample one deterministic network. Same contract as Posterior.sample."""
        return self.layers_for_masks(self.sample_masks(key))

    # ------------------------------------------------------------------
    # Sound route: mask boxes and disjoint partitions
    # ------------------------------------------------------------------

    def mask_box(self, states: MaskStates) -> WeightBox:
        """
        Build a WeightBox from per-unit states (1 kept, 0 dropped, -1 free).

        ``log_prob`` is the exact Bernoulli mass: fixed units contribute
        their keep/drop probability, free units contribute factor 1.
        """
        if len(states) != self.n_mask_layers:
            raise ValueError(
                f"Expected {self.n_mask_layers} state vectors, got {len(states)}."
            )

        box_layers = [(self.W[0], self.W[0], self.b[0], self.b[0])]
        log_prob = 0.0

        for k, st in enumerate(states):
            st = jnp.asarray(st, dtype=jnp.int8)
            if st.shape != (self.mask_dims[k],):
                raise ValueError(
                    f"State vector {k} has shape {st.shape}; expected "
                    f"({self.mask_dims[k]},)."
                )

            kept = (st == 1)[:, None]
            free = (st == -1)[:, None]

            w_scaled = self.W[k + 1] * self.scale
            zero = jnp.zeros_like(w_scaled)

            w_l = jnp.where(
                free, jnp.minimum(w_scaled, 0.0), jnp.where(kept, w_scaled, zero)
            )
            w_u = jnp.where(
                free, jnp.maximum(w_scaled, 0.0), jnp.where(kept, w_scaled, zero)
            )

            b = self.b[k + 1]
            box_layers.append((w_l, w_u, b, b))

            n_kept = int(jnp.sum(st == 1))
            n_drop = int(jnp.sum(st == 0))
            log_prob += n_kept * self._log_keep + n_drop * self._log_drop

        return WeightBox(
            layers=box_layers,
            log_prob=log_prob,
            z_l_flat=None,
            z_u_flat=None,
        )

    def full_support_box(self) -> WeightBox:
        """Single box covering every mask, mass exactly 1. If it certifies,
        P_safe = 1 outright."""
        states = [
            np.full(n, -1, dtype=np.int8) for n in self.mask_dims
        ]
        return self.mask_box(states)
    
    def _flatten_fixed_units(
        self,
        fixed_units: list[np.ndarray],
    ) -> list[tuple[int, int]]:
        """Validate fixed_units and flatten to a list of (layer, unit)."""
        if len(fixed_units) != self.n_mask_layers:
            raise ValueError(
                f"Expected {self.n_mask_layers} index arrays, got "
                f"{len(fixed_units)}."
            )

        flat: list[tuple[int, int]] = []
        for k, idx in enumerate(fixed_units):
            idx = np.asarray(idx, dtype=np.int64).reshape(-1)
            if idx.size != np.unique(idx).size:
                raise ValueError(f"fixed_units[{k}] contains duplicates.")
            if idx.size and (idx.min() < 0 or idx.max() >= self.mask_dims[k]):
                raise ValueError(
                    f"fixed_units[{k}] out of range for width "
                    f"{self.mask_dims[k]}."
                )
            flat.extend((k, int(i)) for i in idx)

        n_cells = 2 ** len(flat)
        if n_cells > self.max_partition_cells:
            raise ValueError(
                f"Partition would have {n_cells} cells "
                f"(> max_partition_cells={self.max_partition_cells}). "
                "Fix fewer units or raise the cap."
            )
        return flat
    
    def partition_states(
        self,
        fixed_units: list[np.ndarray],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Enumerate all assignments over ``fixed_units`` (one index array per
        hidden layer; may be empty), leaving all other units free. The
        resulting cells are pairwise disjoint (any two differ on at least
        one fixed unit) and their masses sum to exactly 1, so

            P_safe >= sum of exp(log_prob) over certified cells

        with no Bonferroni correction needed.

        Returns:
            states: one int8 array of shape (n_cells, width_k) per hidden
                layer (1 kept, 0 dropped, -1 free).
            log_probs: float64 array (n_cells,) of exact Bernoulli log-masses.
        """
        flat = self._flatten_fixed_units(fixed_units)
        n_fixed = len(flat)
        n_cells = 2 ** n_fixed

        # Invariant: bits[c, j] (slot j's assignment in cell c) must match
        # itertools.product((0, 1), repeat=n_fixed) ordering, i.e. slot 0 is
        # most significant. partition_weight_boxes and the test's
        # _cell_states both key off this exact order (see WORKPLAN.md #3).
        cell_ids = np.arange(n_cells, dtype=np.int64)
        shifts = n_fixed - 1 - np.arange(n_fixed, dtype=np.int64)
        bits = ((cell_ids[:, None] >> shifts[None, :]) & 1).astype(np.int8)

        states = [np.full((n_cells, n), -1, dtype=np.int8) for n in self.mask_dims]
        for j, (k, i) in enumerate(flat):
            states[k][:, i] = bits[:, j]

        n_kept = bits.sum(axis=1).astype(np.float64)
        n_drop = float(n_fixed) - n_kept
        log_probs = n_kept * self._log_keep + n_drop * self._log_drop
        return states, log_probs

    def partition_weight_boxes(
        self,
        fixed_units: list[np.ndarray],
    ) -> list[WeightBox]:
        """Same partition as partition_states, materialised as WeightBoxes."""
        states, log_probs = self.partition_states(fixed_units)
        n_cells = int(log_probs.shape[0])

        layer_bounds = [
            self.stacked_mask_layer_bounds(states[k], k)
            for k in range(self.n_mask_layers)
        ]

        boxes = []
        for c in range(n_cells):
            box_layers = [(self.W[0], self.W[0], self.b[0], self.b[0])]
            for k in range(self.n_mask_layers):
                w_l, w_u = layer_bounds[k]
                b = self.b[k + 1]
                box_layers.append((w_l[c], w_u[c], b, b))
            boxes.append(
                WeightBox(
                    layers=box_layers,
                    log_prob=float(log_probs[c]),
                    z_l_flat=None,
                    z_u_flat=None,
                )
            )
        return boxes

    def stacked_mask_layer_bounds(
        self,
        states_k: np.ndarray,
        k: int,
    ) -> tuple[Array, Array]:
        """
        Elementwise weight bounds of masked layer k for every cell at once.

        Args:
            states_k: (n_cells, width_k) int8 state matrix for hidden layer k.
            k: hidden-layer index (acts on rows of W[k + 1]).

        Returns:
            (w_l, w_u), each (n_cells, n_in, n_out).
        """
        st = jnp.asarray(states_k, dtype=jnp.int8)[:, :, None]
        w_scaled = (self.W[k + 1] * self.scale)[None, :, :]

        kept = st == 1
        free = st == -1
        zero = jnp.zeros_like(w_scaled)

        w_l = jnp.where(
            free, jnp.minimum(w_scaled, 0.0), jnp.where(kept, w_scaled, zero)
        )
        w_u = jnp.where(
            free, jnp.maximum(w_scaled, 0.0), jnp.where(kept, w_scaled, zero)
        )
        return w_l, w_u

    # ------------------------------------------------------------------
    # Heuristic choice of the fixed set F
    # ------------------------------------------------------------------

    def activation_upper_bounds(self, x_L: Array, x_U: Array) -> list[Array]:
        """
        Sound per-unit upper bounds on the *scaled* hidden activations
        (h_i / (1 - p)) over the input box, under the all-free relaxation.
        Used only to rank units; soundness of certification never depends
        on this.
        """
        h_L = jnp.asarray(x_L).reshape(-1)
        h_U = jnp.asarray(x_U).reshape(-1)

        uppers = []
        for k in range(self.n_mask_layers):
            W, b = self.W[k], self.b[k]
            mu = (h_U + h_L) / 2.0
            r = (h_U - h_L) / 2.0
            pre_U = mu @ W + r @ jnp.abs(W) + b
            a_U = jax.nn.relu(pre_U) * self.scale
            uppers.append(a_U)
            # All-free relaxation for the next layer's input interval.
            h_L = jnp.zeros_like(a_U)
            h_U = a_U
        return uppers

    def choose_fixed_units(
        self,
        x_L: Array,
        x_U: Array,
        n_fixed: int | list[int],
    ) -> list[np.ndarray]:
        """
        Rank units by a swing heuristic  u_i * ||W_next[i, :]||_1  (the
        largest output perturbation a free mask on unit i can induce) and fix
        the top ``n_fixed`` per hidden layer.
        """
        if isinstance(n_fixed, int):
            n_fixed = [n_fixed] * self.n_mask_layers
        if len(n_fixed) != self.n_mask_layers:
            raise ValueError(
                f"n_fixed must be an int or a list of length "
                f"{self.n_mask_layers}."
            )

        uppers = self.activation_upper_bounds(x_L, x_U)

        fixed = []
        for k, a_U in enumerate(uppers):
            row_l1 = jnp.sum(jnp.abs(self.W[k + 1]), axis=1)
            score = np.asarray(a_U * row_l1)
            n_k = int(min(n_fixed[k], self.mask_dims[k]))
            top = np.argsort(-score)[:n_k]
            fixed.append(np.sort(top))
        return fixed
    
