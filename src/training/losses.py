from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

Array = jax.Array


def ce_from_logits(logits: Array, labels: Array) -> Array:
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=labels,
    ).mean()


def nll_from_probs(probs: Array, labels: Array, eps: float = 1e-8) -> Array:
    labels = labels.astype(jnp.int32)
    idx = jnp.arange(labels.shape[0])
    p_y = probs[idx, labels]
    return -jnp.mean(jnp.log(jnp.clip(p_y, eps, 1.0)))


def robust_likelihood_nll(
    *,
    clean_logits: Array,
    logit_l: Array,
    logit_u: Array,
    labels: Array,
    rob_lam: float,
):
    """
    Implements Wicker-style two-point robust likelihood:
 
        p_rob(y|x,w)
          = rob_lam * p_y(x,w)
            + (1 - rob_lam) * p_y^IBP([x-eps,x+eps],w)
 
    where the IBP worst-case true-class softmax uses:
      - lower logit bound for the true class
      - upper logit bounds for all non-true classes
    TODO: Clean up below
    """
    labels = labels.astype(jnp.int32)
    idx = jnp.arange(labels.shape[0])
    y_onehot = jax.nn.one_hot(labels, clean_logits.shape[-1])
 
    worst_logits = y_onehot * logit_l + (1.0 - y_onehot) * logit_u
    clean_logp_y = jax.nn.log_softmax(clean_logits, axis=-1)[idx, labels]   # (B,)
    worst_logp_y = jax.nn.log_softmax(worst_logits, axis=-1)[idx, labels]   # (B,)
    """
    clean_probs = jax.nn.softmax(clean_logits, axis=-1)
    worst_probs = jax.nn.softmax(worst_logits, axis=-1)
 
    robust_probs = (
        rob_lam * clean_probs
        + (1.0 - rob_lam) * worst_probs
    )
 
    robust_nll = nll_from_probs(robust_probs, labels)"""

    log_lam = jnp.log(jnp.clip(rob_lam, 1e-30, 1.0))
    log_1m_lam = jnp.log(jnp.clip(1.0 - rob_lam, 1e-30, 1.0))

    stacked = jnp.stack([
        log_lam + clean_logp_y,
        log_1m_lam + worst_logp_y,
        ], axis=0)  
    robust_logp_y = jax.nn.logsumexp(stacked, axis=0)                       # (B,)
    robust_nll = -jnp.mean(robust_logp_y)

    clean_nll = -jnp.mean(clean_logp_y)     # identical to ce_from_logits
    worst_nll = -jnp.mean(worst_logp_y)    

    resp = jnp.mean(jnp.exp(log_1m_lam + worst_logp_y - robust_logp_y))
 
    """
    log_p = jax.nn.logsumexp(jnp.stack([
        jnp.log(rob_lam) + clean_logp[y_i],
        jnp.log1p(-rob_lam) + worst_logp[y_i],
    ]))
    robust_nll = -log_p
    clean_nll = ce_from_logits(clean_logits, labels)
    worst_nll = nll_from_probs(worst_probs, labels)
    """
    return robust_nll, {
        "clean_nll": clean_nll,
        "worst_nll": worst_nll,
        "worst_logits": worst_logits,
        "resp": resp,
    }
 
 
def make_mc_dropout_loss(*, robust_train=False):
    def loss_fn(model, batch, rngs, epsilon, rob_lam=1.0):
        x, y = batch["image"], batch["label"]

        if robust_train:
            clean_logits, logit_l, logit_u = model.clean_and_ibp_logits(
                x,
                epsilon=epsilon,
                rngs=rngs,
                clip_min=0.0,
                clip_max=1.0,
            )

            nll, rob_aux = robust_likelihood_nll(
                clean_logits=clean_logits,
                logit_l=logit_l,
                logit_u=logit_u,
                labels=y,
                rob_lam=rob_lam,
            )

            return nll, {
                "logits": clean_logits,
                "nll": nll,
                **rob_aux,
            }

        logits = model(
            x,
            disable_dropout=False,
            return_logits=True,
            rngs=rngs,
        )
        nll = ce_from_logits(logits, y)

        return nll, {
            "logits": logits,
            "nll": nll,
            "clean_nll": nll,
            "worst_nll": jnp.array(0.0),
            "worst_logits": logits,
        }

    return loss_fn


def make_bbb_loss(
    *,
    n_train: int,
    beta: float = 1.0,
    robust_train: bool = False,
) -> Callable:
    def bbb_loss(model, batch, rngs: nnx.Rngs, epsilon=0.0, rob_lam=1.0):  
        x = batch["image"]
        y = batch["label"]
 
        if robust_train:
            clean_logits, logit_l, logit_u, kl = model.clean_and_ibp_logits(
                x,
                epsilon=epsilon,
                rngs=rngs,
                sample=True,
                clip_min=0.0,
                clip_max=1.0,
            )
 
            nll, rob_aux = robust_likelihood_nll(
                clean_logits=clean_logits,
                logit_l=logit_l,
                logit_u=logit_u,
                labels=y,
                rob_lam=rob_lam,
            )
 
            logits_for_metrics = clean_logits
 
            aux_extra = {
                "clean_nll": rob_aux["clean_nll"],
                "worst_nll": rob_aux["worst_nll"],
                "worst_logits": rob_aux["worst_logits"],
                "resp": rob_aux["resp"],
            }
 
        else:
            logits, kl = model(
                x,
                rngs=rngs,
                sample=True,
                return_logits=True,
                return_kl=True,
            )
 
            nll = ce_from_logits(logits, y)
            logits_for_metrics = logits
 
            aux_extra = {
                "clean_nll": nll,
                "worst_nll": jnp.array(0.0),
                "worst_logits": logits,
            }
 
        kl_scaled = beta * kl / n_train
        loss = nll + kl_scaled
 
        return loss, {
            "logits": logits_for_metrics,
            "nll": nll,
            "kl": kl,
            "kl_scaled": kl_scaled,
            **aux_extra,
        }
 
    return bbb_loss

def make_eval_loss(model_type: str):
    def eval_loss(model, batch, rngs: nnx.Rngs | None = None):
        if model_type in ("bbb", "vogn"):
            logits = model(
                batch["image"],
                sample=False,
                return_logits=True,
            )

        elif model_type == "mc_dropout":
            logits = model(
                batch["image"],
                disable_dropout=True,
                return_logits=True,
            )

        else:
            logits = model(
                batch["image"],
                return_logits=True,
            )

        loss = ce_from_logits(logits, batch["label"])

        return loss, {
            "logits": logits,
            "nll": loss,
        }

    return eval_loss