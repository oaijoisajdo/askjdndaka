from __future__ import annotations
from typing import Literal
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, log_ndtr

Array = jax.Array

BoxGeneration = Literal["sample", "centered"]

class GaussianPosterior:
    def __init__(
        self,
        model,
        *,
        min_sigma: float = 1e-8,
        store_z_bounds: bool = True,
    ):
        self.model = model
        self.min_sigma = min_sigma
        self.store_z_bounds = store_z_bounds

        # Store posterior params for every weight and bias 
        # N(mu, sigma) at every param

        self.mu_w = []
        self.sigma_w = []
        self.mu_b = []
        self.sigma_b = []

        for layer in self._iter_bayesian_layers():
            mu_w, sigma_w, mu_b, sigma_b = self._layer_posterior_params(
                layer,
                min_sigma=min_sigma,
            )

            self.mu_w.append(mu_w)
            self.sigma_w.append(sigma_w)
            self.mu_b.append(mu_b)
            self.sigma_b.append(sigma_b)

        self.n_layers = len(self.mu_w)

    def _iter_bayesian_layers(self):
        for i in range(self.model.depth):
            yield self.model.layers[i]
        yield self.model.output_layer

    @staticmethod
    def _as_array(x):
        # NNX Param / Variable -> underlying JAX array
        if hasattr(x, "value"):
            return jnp.asarray(x.value)
        return jnp.asarray(x)

    @classmethod
    def _layer_posterior_params(cls, layer, *, min_sigma: float = 1e-8):
        mu_w = cls._as_array(layer.mu_w)
        mu_b = cls._as_array(layer.mu_b)

        if getattr(layer, "MODEL_TYPE", None) == "bbb":
            sigma_w = jax.nn.softplus(cls._as_array(layer.rho_w))
            sigma_b = jax.nn.softplus(cls._as_array(layer.rho_b))

        elif getattr(layer, "MODEL_TYPE", None) == "vogn":
            sigma_w, sigma_b = layer.sigma()
            sigma_w = jnp.asarray(sigma_w)
            sigma_b = jnp.asarray(sigma_b)

        else:
            raise TypeError(
                "Posterior only supports BBBLinear and VOGNLinear layers. "
                f"Got layer type {type(layer).__name__}."
            )

        sigma_w = jnp.maximum(sigma_w, min_sigma)
        sigma_b = jnp.maximum(sigma_b, min_sigma)

        return mu_w, sigma_w, mu_b, sigma_b


    def sample(self, key: Array, *, return_sigma: bool = False, return_z: bool = False) -> list[tuple]:
        """
        Sample one deterministic network from the VI posterior.

        Returns:
            [(w0, b0), (w1, b1), ...]
        """
        keys = jax.random.split(key, 2 * self.n_layers)

        samples = []

        for i in range(self.n_layers):
            key_w = keys[2 * i]
            key_b = keys[2 * i + 1]

            # Sample Gaussian noise (z ~ N(0,1)) for reparametrization
            z_w = jax.random.normal(key_w, shape=self.mu_w[i].shape)
            z_b = jax.random.normal(key_b, shape=self.mu_b[i].shape)

            # theta = mu + sigma * z 
            w = self.mu_w[i] + self.sigma_w[i] * z_w
            b = self.mu_b[i] + self.sigma_b[i] * z_b

            if return_sigma and return_z:
                samples.append((w, b, self.sigma_w[i], self.sigma_b[i], z_w, z_b))
            elif return_sigma:
                samples.append((w, b, self.sigma_w[i], self.sigma_b[i]))
            elif return_z:
                samples.append((w, b, z_w, z_b))
            else:
                samples.append((w, b))

        return samples
    
    def make_weight_boxes(
        self,
        *,
        key: Array | None,
        n_samples: int,
        gamma: float,
        box_generation: BoxGeneration = "sample",
    ) -> list[WeightBox]:
        
        if box_generation == "sample":
            if key is None:
                raise ValueError("key is required when box_generation='sample'.")

            keys = jax.random.split(key, n_samples)
            return [self.sample_weight_box(k, gamma) for k in keys]

        if box_generation == "centered":
            # n_samples is intentionally ignored here: one centered box.
            return [self.centered_weight_box(gamma)]

        raise ValueError(f"Unknown box_generation: {box_generation}")

    def sample_weight_box(self, key: Array, gamma: float) -> WeightBox:

        """
        Constructs sampled weight boxes
        """
        sampled = self.sample(key, return_z=True)

        z_w_l_layers = []
        z_w_u_layers = []
        z_b_l_layers = []
        z_b_u_layers = []

        for _, _, z_w, z_b in sampled:
            z_w_l_layers.append(z_w - gamma)
            z_w_u_layers.append(z_w + gamma)
            z_b_l_layers.append(z_b - gamma)
            z_b_u_layers.append(z_b + gamma)

        return self.weight_box_from_z_bounds(
            z_w_l_layers,
            z_w_u_layers,
            z_b_l_layers,
            z_b_u_layers,
        )
    def centered_weight_box(self, gamma: float) -> WeightBox:
        z_w_l_layers = []
        z_w_u_layers = []
        z_b_l_layers = []
        z_b_u_layers = []

        for i in range(self.n_layers):
            z_w_l_layers.append(jnp.full_like(self.mu_w[i], -gamma))
            z_w_u_layers.append(jnp.full_like(self.mu_w[i], gamma))
            z_b_l_layers.append(jnp.full_like(self.mu_b[i], -gamma))
            z_b_u_layers.append(jnp.full_like(self.mu_b[i], gamma))

        return self.weight_box_from_z_bounds(
            z_w_l_layers,
            z_w_u_layers,
            z_b_l_layers,
            z_b_u_layers,
    )
    
    def weight_box_from_z_bounds(
        self,
        z_w_l_layers,
        z_w_u_layers,
        z_b_l_layers,
        z_b_u_layers,
    ) -> WeightBox:
        box_layers = []
        z_l_parts = []
        z_u_parts = []
        log_prob = jnp.array(0.0)

        for i in range(self.n_layers):
            z_w_l = z_w_l_layers[i]
            z_w_u = z_w_u_layers[i]
            z_b_l = z_b_l_layers[i]
            z_b_u = z_b_u_layers[i]

            w_l = self.mu_w[i] + self.sigma_w[i] * z_w_l
            w_u = self.mu_w[i] + self.sigma_w[i] * z_w_u
            b_l = self.mu_b[i] + self.sigma_b[i] * z_b_l
            b_u = self.mu_b[i] + self.sigma_b[i] * z_b_u

            box_layers.append((w_l, w_u, b_l, b_u))

            log_prob += self.log_standard_normal_interval_prob(z_w_l, z_w_u)
            log_prob += self.log_standard_normal_interval_prob(z_b_l, z_b_u)

            z_l_parts.append(z_w_l.reshape(-1))
            z_l_parts.append(z_b_l.reshape(-1))
            z_u_parts.append(z_w_u.reshape(-1))
            z_u_parts.append(z_b_u.reshape(-1))

        return WeightBox(
            layers=box_layers,
            log_prob=log_prob,
            z_l_flat=jnp.concatenate(z_l_parts),
            z_u_flat=jnp.concatenate(z_u_parts),
        )


    @staticmethod
    def log_standard_normal_interval_prob_non_mirrored(z_l: Array, z_u: Array) -> Array:
        """
        log Π_i [Φ(z_u_i) - Φ(z_l_i)]
        """
        probs = ndtr(z_u) - ndtr(z_l)
        tiny = jnp.finfo(probs.dtype).tiny
        return jnp.sum(jnp.log(jnp.clip(probs, tiny, 1.0)))
    
    @staticmethod
    def log_standard_normal_interval_prob(
        z_l: Array,
        z_u: Array,
    ) -> Array:
        """
        Compute log Π_i [Φ(z_u_i) - Φ(z_l_i)] stably.

        Empty or reversed component intervals give total probability zero
        and therefore log probability -inf.
        """
        # Reflect intervals wholly in the positive half-line.
        reflect = z_l >= 0
        a = jnp.where(reflect, -z_u, z_l)
        b = jnp.where(reflect, -z_l, z_u)

        log_cdf_a = log_ndtr(a)
        log_cdf_b = log_ndtr(b)

        # For valid intervals, delta <= 0. Clamping protects against a
        # tiny positive value caused by floating-point roundoff.
        delta = jnp.minimum(log_cdf_a - log_cdf_b, 0.0)

        # log(exp(log_cdf_b) - exp(log_cdf_a))
        # = log_cdf_b + log(1 - exp(delta)).
        # -expm1(delta) is accurate when delta is close to zero.
        log_mass = log_cdf_b + jnp.log(-jnp.expm1(delta))

        # Also handles equal infinite endpoints, where inf-inf above is NaN.
        log_mass = jnp.where(z_u <= z_l, -jnp.inf, log_mass)

        return jnp.sum(log_mass)

class WeightBox:
    def __init__(
        self,
        layers: list[tuple[Array, Array, Array, Array]],
        log_prob: Array,
        z_l_flat: Array | None = None,
        z_u_flat: Array | None = None,
    ):
        """
        layers stores:
            [(w_l, w_u, b_l, b_u), ...]
        """
        self.layers = layers
        self.log_prob = log_prob
        self.z_l_flat = z_l_flat
        self.z_u_flat = z_u_flat

def make_deterministic_box_from_layers(
    layers: list[tuple[Array, Array]],
) -> WeightBox:
    """
    Wrap sampled deterministic weights as a zero-width WeightBox.

    A zero-width box makes propagate_IBP collapse to the exact point
    forward pass (all interval radii are zero), so no separate forward
    implementation is needed and the architecture is defined in exactly
    one place (bound_propagation.py).
    """
    box_layers = [(w, w, b, b) for (w, b) in layers]

    return WeightBox(
        layers=box_layers,
        log_prob=jnp.array(-jnp.inf),  # not used in the statistical estimator
        z_l_flat=None,
        z_u_flat=None,
    )