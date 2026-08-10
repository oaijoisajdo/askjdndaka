from __future__ import annotations

from typing import Literal, ClassVar
import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx

Array = jax.Array
PriorType = Literal["gaussian", "scale_mixture"]


@dataclass(frozen=True)
class PriorConfig:
    name: PriorType = "gaussian"

    # Gaussian
    mu: float = 0.0
    sigma: float = 0.2

    # Scale mixture
    pi: float = 0.5
    sigma1: float = 1.0
    sigma2: float = 0.002

    @property
    def precision(self) -> float:
        return 1.0 / (self.sigma ** 2)




# ---------------------------------------------------------------------------
# Shared layer interface
# ---------------------------------------------------------------------------

class BayesianLinearBase(nnx.Module):
    """
    Contract both BBBLinear and VOGNLinear implement. VI_BNN, dense_ibp
    usage, and clean_and_ibp_logits only ever call these three, so they
    don't need to know which algorithm a given layer is using.

    MODEL_TYPE is how VI_BNN.model_type / get_config figure out what kind
    of model they're attached to -- set it on each concrete subclass.
    """

    MODEL_TYPE: ClassVar[str] = "base"

    d_in: int
    d_out: int

    def sample_params(self, *, rngs: nnx.Rngs | None = None, sample: bool = True):
        raise NotImplementedError

    def apply_params(self, x: Array, w: Array, b: Array) -> Array:
        return x @ w + b

    def kl(self, w: Array | None = None, b: Array | None = None) -> Array:
        """Default: no explicit KL term. BBBLinear overrides this."""
        return jnp.array(0.0)
    def __call__(
        self,
        x: Array,
        *,
        rngs: nnx.Rngs | None = None,
        sample: bool = True,
        return_kl: bool = False,
    ):
        w, b = self.sample_params(rngs=rngs, sample=sample)
        out = self.apply_params(x, w, b)

        if return_kl:
            return out, self.kl(w, b)

        return out


# ---------------------------------------------------------------------------
# BBB
# ---------------------------------------------------------------------------

class BBBLinear(BayesianLinearBase):
    """Bayes-by-Backprop linear layer (renamed from BayesianLinear)."""

    MODEL_TYPE: ClassVar[str] = "bbb"

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        rngs: nnx.Rngs,
        prior: PriorConfig = PriorConfig(),
        rho_init: float = -3.0,
        init: str = "xavier",
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.prior = prior

        key_w_mu = rngs.params()
        self.mu_w = self.initializer(init, key_w_mu)
        self.rho_w = nnx.Param(jnp.ones((d_in, d_out)) * rho_init)

        self.mu_b = nnx.Param(jnp.zeros((d_out,)))
        self.rho_b = nnx.Param(jnp.ones(d_out,) * rho_init)

    def __call__(
        self,
        x: Array,
        *,
        rngs: nnx.Rngs | None = None,
        sample: bool = True,
        return_kl: bool = False,
    ):
        w, b = self.sample_params(rngs=rngs, sample=sample)
        out = self.apply_params(x, w, b)

        if return_kl:
            kl = self.kl(w, b)
            return out, kl

        return out

    def kl(self, w: Array | None = None, b: Array | None = None) -> Array:
        w_sigma = jax.nn.softplus(self.rho_w[...])
        b_sigma = jax.nn.softplus(self.rho_b[...])

        if self.prior.name == "gaussian":
            prior_mu = self.prior.mu
            prior_sigma = self.prior.sigma

            kl_w = jnp.sum(
                jnp.log(prior_sigma / w_sigma)
                + (w_sigma**2 + (self.mu_w[...] - prior_mu) ** 2)
                / (2.0 * prior_sigma**2)
                - 0.5
            )

            kl_b = jnp.sum(
                jnp.log(prior_sigma / b_sigma)
                + (b_sigma**2 + (self.mu_b[...] - prior_mu) ** 2)
                / (2.0 * prior_sigma**2)
                - 0.5
            )

            return kl_w + kl_b

        if self.prior.name == "scale_mixture":
            if w is None or b is None:
                raise ValueError("Scale-mixture KL requires sampled w and b.")

            log_q = (
                gaussian_log_prob(w, self.mu_w[...], w_sigma)
                + gaussian_log_prob(b, self.mu_b[...], b_sigma)
            )

            log_p = (
                scale_mixture_log_prob(w, self.prior)
                + scale_mixture_log_prob(b, self.prior)
            )

            return log_q - log_p

        raise ValueError(f"Unknown prior: {self.prior.name}")

    def initializer(self, init, key_w_mu):
        if init == "xavier":
            limit = jnp.sqrt(6.0 / (self.d_in + self.d_out))
            mu_w = nnx.Param(jax.random.uniform(
                key=key_w_mu, shape=(self.d_in, self.d_out),
                minval=-limit, maxval=limit,
            ))
        if init == "nitarshan":
            mu_w = nnx.Param(jax.random.uniform(
                key=key_w_mu, shape=(self.d_in, self.d_out),
                minval=-0.2, maxval=0.2,
            ))
        if init == "normal":
            mu_w = nnx.Param(jax.random.normal(
                key=key_w_mu, shape=(self.d_in, self.d_out),
            ))
        return mu_w

    def sample_params(self, *, rngs: nnx.Rngs | None = None, sample: bool = True):
        sigma_w = jax.nn.softplus(self.rho_w[...])
        sigma_b = jax.nn.softplus(self.rho_b[...])

        if sample:
            if rngs is None:
                raise ValueError("rngs must be provided when sample=True")

            key_w = rngs.bayes()
            key_b = rngs.bayes()

            eps_w = jax.random.normal(key_w, (self.d_in, self.d_out))
            eps_b = jax.random.normal(key_b, (self.d_out,))

            w = self.mu_w[...] + sigma_w * eps_w
            b = self.mu_b[...] + sigma_b * eps_b
        else:
            w = self.mu_w[...]
            b = self.mu_b[...]

        return w, b


# ---------------------------------------------------------------------------
# VOGN
# ---------------------------------------------------------------------------

def _xavier_init(key, d_in, d_out):
    limit = jnp.sqrt(6.0 / (d_in + d_out))
    return jax.random.uniform(key, (d_in, d_out), minval=-limit, maxval=limit)


class VOGNLinear(BayesianLinearBase):
    """
    Variational Online Gauss-Newton linear layer. Gaussian prior only.

    mu_w/mu_b: trainable nnx.Param, but updated manually in
    make_vogn_train_step -- never via nnx.Optimizer/optax.
    s_w/s_b:   running Gauss-Newton (Fisher diag) estimate -> sigma.
    m_w/m_b:   running gradient momentum.
    s_w/s_b/m_w/m_b are nnx.Variable (not nnx.Param): they must never
    receive an autodiff/ELBO gradient themselves.

    dataset_size is fixed at construction -- see the note in vogn_step.py
    about why VOGN needs N earlier than BBB does.
    """

    MODEL_TYPE: ClassVar[str] = "vogn"

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        rngs: nnx.Rngs,
        prior: PriorConfig = PriorConfig(),
        dataset_size: int,
        s_init: float = 1.0,
        temperature: float = 1.0,
        init: str = "xavier",
    ):
        if prior.name != "gaussian":
            raise ValueError("VOGNLinear only supports Gaussian priors.")

        self.d_in = d_in
        self.d_out = d_out
        self.prior = prior
        self.dataset_size = dataset_size
        self.temperature = temperature
        key_w = rngs.params()
        if init == "xavier":
            mu_w0 = _xavier_init(key_w, d_in, d_out)
        elif init == "normal":
            mu_w0 = jax.random.normal(key_w, (d_in, d_out))
        else:
            raise ValueError(f"Unknown init: {init}")

        self.mu_w = nnx.Param(mu_w0)
        self.mu_b = nnx.Param(jnp.zeros((d_out,)))

        self.s_w = nnx.Variable(jnp.ones((d_in, d_out)) * s_init)
        self.s_b = nnx.Variable(jnp.ones((d_out,)) * s_init)
        self.m_w = nnx.Variable(jnp.zeros((d_in, d_out)))
        self.m_b = nnx.Variable(jnp.zeros((d_out,)))
        self.s_ema_weight = nnx.Variable(jnp.array(0.0))
        self.m_ema_weight = nnx.Variable(jnp.array(0.0))

    def sigma_prev(self) -> tuple[Array, Array]:
        prec = 1.0 / (self.prior.sigma ** 2)
        sigma_w = 1.0 / jnp.sqrt(self.dataset_size * self.s_w[...] + prec)
        sigma_b = 1.0 / jnp.sqrt(self.dataset_size * self.s_b[...] + prec)
        return sigma_w, sigma_b
    
    def sigma(self) -> tuple[Array, Array]:
        prec = 1.0 / (self.prior.sigma**2)

        correction = self.s_ema_weight[...]
        safe_correction = jnp.maximum(
            correction,
            jnp.asarray(1e-12, dtype=self.s_w[...].dtype),
        )

        # Before the first curvature update, use prior-only uncertainty.
        s_w_hat = jnp.where(
            correction > 0.0,
            self.s_w[...] / safe_correction,
            jnp.zeros_like(self.s_w[...]),
        )
        s_b_hat = jnp.where(
            correction > 0.0,
            self.s_b[...] / safe_correction,
            jnp.zeros_like(self.s_b[...]),
        )

        sigma_w = 1.0 / jnp.sqrt(self.n_eff * s_w_hat + prec)
        sigma_b = 1.0 / jnp.sqrt(self.n_eff * s_b_hat + prec)

        return sigma_w, sigma_b
    
    def sample_params(self, *, rngs: nnx.Rngs | None = None, sample: bool = True):
        if sample:
            if rngs is None:
                raise ValueError("rngs must be provided when sample=True")
            sigma_w, sigma_b = self.sigma()
            eps_w = jax.random.normal(rngs.bayes(), (self.d_in, self.d_out))
            eps_b = jax.random.normal(rngs.bayes(), (self.d_out,))
            w = self.mu_w[...] + sigma_w * eps_w
            b = self.mu_b[...] + sigma_b * eps_b
        else:
            w = self.mu_w[...]
            b = self.mu_b[...]
        return w, b
    
    @property
    def n_eff(self) -> float:
        return self.dataset_size / self.temperature

# ---------------------------------------------------------------------------
# VI_BNN
# ---------------------------------------------------------------------------

class VI_BNN(nnx.Module):

    def __init__(
        self,
        d_in: int = 784,
        d_out: int = 10,
        width: int = 24,
        depth: int = 1,
        *,
        rngs: nnx.Rngs,
        prior: PriorConfig = PriorConfig(),
        layer_cls: type[BayesianLinearBase] = BBBLinear,   # NEW
        layer_kwargs: dict | None = None,                    # NEW
    ):
        layer_kwargs = layer_kwargs or {}

        self.d_in = d_in
        self.d_out = d_out
        self.width = width
        self.depth = depth
        self.prior = prior
        self.layer_cls = layer_cls          # static metadata, like self.prior
        self.layer_kwargs = layer_kwargs    # static metadata

        self.layers = nnx.List([
            layer_cls(
                d_in if i == 0 else width, width,
                rngs=rngs, prior=prior, **layer_kwargs,
            )
            for i in range(depth)
        ])
        self.output_layer = layer_cls(
            width, d_out, rngs=rngs, prior=prior, **layer_kwargs,
        )

    def __call__(
        self,
        x: Array,
        *,
        rngs: nnx.Rngs | None = None,
        sample: bool = True,
        return_kl: bool = False,
        return_logits: bool = False,
    ):
        total_kl = jnp.array(0.0)

        for layer in self.layers:
            if return_kl:
                x, kl = layer(x, rngs=rngs, sample=sample, return_kl=True)
                total_kl += kl
            else:
                x = layer(x, rngs=rngs, sample=sample)

            x = nnx.relu(x)

        if return_kl:
            logits, kl = self.output_layer(
                x, rngs=rngs, sample=sample, return_kl=True
            )
            total_kl += kl
        else:
            logits = self.output_layer(x, rngs=rngs, sample=sample)

        output = logits if return_logits else nnx.softmax(logits)

        if return_kl:
            return output, total_kl

        return output

    def clean_and_ibp_logits(
        self,
        x: Array,
        *,
        epsilon: float,
        rngs: nnx.Rngs,
        sample: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        """
        Samples one network and computes:
        clean_logits = f_w(x)
        logit_l, logit_u = IBP bounds over [x-eps, x+eps]
        total_kl = KL for the same sampled weights (0.0 for VOGN layers,
        since VOGN has no explicit KL term -- see BayesianLinearBase.kl)

        Assumes flattened image inputs in [clip_min, clip_max].
        """
        clean = x

        x_l = jnp.clip(x - epsilon, clip_min, clip_max)
        x_u = jnp.clip(x + epsilon, clip_min, clip_max)

        total_kl = jnp.array(0.0)

        for layer in self.layers:
            w, b = layer.sample_params(rngs=rngs, sample=sample)
            total_kl += layer.kl(w, b)

            clean = nnx.relu(layer.apply_params(clean, w, b))

            x_l, x_u = dense_ibp(x_l, x_u, w, b)
            x_l = nnx.relu(x_l)
            x_u = nnx.relu(x_u)

        w, b = self.output_layer.sample_params(rngs=rngs, sample=sample)
        total_kl += self.output_layer.kl(w, b)

        clean_logits = self.output_layer.apply_params(clean, w, b)
        logit_l, logit_u = dense_ibp(x_l, x_u, w, b)

        return clean_logits, logit_l, logit_u, total_kl

    @property
    def model_type(self) -> str:
        return self.layer_cls.MODEL_TYPE  # CHANGED: generic, not hardcoded "bbb"

    def get_config(self) -> dict:
        return {
            "model_type": self.model_type,
            "d_in": self.d_in,
            "d_out": self.d_out,
            "width": self.width,
            "depth": self.depth,
            "prior": dataclasses.asdict(self.prior),
            **self.layer_kwargs,  # rho_init for bbb; dataset_size/s_init for vogn
        }


def gaussian_log_prob(x: Array, mu: Array, sigma: Array) -> Array:
    return jnp.sum(
        -0.5 * jnp.log(2.0 * jnp.pi)
        - jnp.log(sigma)
        - 0.5 * ((x - mu) / sigma) ** 2
    )


def scale_mixture_log_prob(x: Array, prior: PriorConfig) -> Array:
    log_p1 = (
        jnp.log(prior.pi)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - jnp.log(prior.sigma1)
        - 0.5 * (x / prior.sigma1) ** 2
    )

    log_p2 = (
        jnp.log1p(-prior.pi)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - jnp.log(prior.sigma2)
        - 0.5 * (x / prior.sigma2) ** 2
    )

    return jnp.sum(jnp.logaddexp(log_p1, log_p2))

def dense_ibp(x_l: Array, x_u: Array, w: Array, b: Array):
    """
    Deterministic IBP through y = x @ w + b.

    x_l, x_u: [batch, d_in]
    w:        [d_in, d_out]
    b:        [d_out]
    """
    w_pos = jnp.maximum(w, 0.0)
    w_neg = jnp.minimum(w, 0.0)

    y_l = x_l @ w_pos + x_u @ w_neg + b
    y_u = x_u @ w_pos + x_l @ w_neg + b
    return y_l, y_u
