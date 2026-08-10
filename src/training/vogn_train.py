"""
vogn_step.py

VOGN training step, built to slot into your existing train.py loop
alongside make_train_step (bbb/mc_dropout/deterministic).

Key structural difference from make_train_step: there is no nnx.Optimizer
here. mu_w/mu_b updates happen by direct assignment inside this function,
coupled to the s_w/s_b/m_w/m_b state also updated here. So this returns a
step function with signature (model, metrics, batch, rngs, step) --
no `optimizer` argument.

Depends on VOGNLinear from vogn_bnn.py (GaussianPrior, dataset_size stored
per-layer at construction time -- see note at bottom).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import flax.nnx as nnx

def make_vogn_train_step(
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    robust_train: bool = False,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Callable:
    @nnx.jit
    def vogn_train_step(
        model,
        metrics: nnx.MultiMetric,
        batch,
        rngs: nnx.Rngs,
        step: int,  # 1-indexed, for Adam-style bias correction
        epsilon: float = 0.0,
        rob_lam: float = 1.0,
    ):
        x, y = batch["image"], batch["label"]
        layers = list(model.layers) + [model.output_layer]

        # One shared weight sample per layer for the whole minibatch
        # (Khan et al. 2018, Algorithm 1 -- MC=1 per step).
        sampled = [layer.sample_params(rngs=rngs, sample=True) for layer in layers]

        def dense_ibp_local(x_l, x_u, w, b):
            w_pos = jnp.maximum(w, 0.0)
            w_neg = jnp.minimum(w, 0.0)
            y_l = x_l @ w_pos + x_u @ w_neg + b
            y_u = x_u @ w_pos + x_l @ w_neg + b
            return y_l, y_u


        def per_example(params, x_i, y_i):
            y_i = y_i.astype(jnp.int32)

            h = x_i

            if robust_train:
                x_l = jnp.clip(x_i - epsilon, clip_min, clip_max)
                x_u = jnp.clip(x_i + epsilon, clip_min, clip_max)

            for (w, b), layer in zip(params[:-1], layers[:-1]):
                h = nnx.relu(layer.apply_params(h, w, b))

                if robust_train:
                    x_l, x_u = dense_ibp_local(x_l, x_u, w, b)
                    x_l = nnx.relu(x_l)
                    x_u = nnx.relu(x_u)

            w, b = params[-1]
            clean_logits = layers[-1].apply_params(h, w, b)

            clean_logp = jax.nn.log_softmax(clean_logits)
            clean_nll = -clean_logp[y_i]

            if not robust_train:
                return clean_nll, {
                    "logits": clean_logits,
                    "clean_nll": clean_nll,
                    "worst_nll": jnp.array(0.0),
                    "nll": clean_nll,
                }

            logit_l, logit_u = dense_ibp_local(x_l, x_u, w, b)

            y_onehot = jax.nn.one_hot(y_i, clean_logits.shape[-1])
            worst_logits = y_onehot * logit_l + (1.0 - y_onehot) * logit_u
            
            worst_logp = jax.nn.log_softmax(worst_logits)
            worst_nll = -worst_logp[y_i]
            #TODO Clean below
            """ 
            clean_probs = jax.nn.softmax(clean_logits)
            worst_probs = jax.nn.softmax(worst_logits)

                  
            robust_p_y = (
                rob_lam * clean_probs[y_i]
                + (1.0 - rob_lam) * worst_probs[y_i]
            )

            #robust_nll = -jnp.log(jnp.clip(robust_p_y, 1e-8, 1.0))
            robust_nll = -jnp.log(robust_p_y + 1e-8)"""

            log_lam = jnp.log(jnp.clip(rob_lam, 1e-30, 1.0))
            log_1m_lam = jnp.log(jnp.clip(1.0 - rob_lam, 1e-30, 1.0))
            terms = jnp.stack([
                log_lam + clean_logp[y_i],
                log_1m_lam + worst_logp[y_i],
            ])
            robust_nll = -jax.nn.logsumexp(terms)
            worst_nll = -jax.nn.log_softmax(worst_logits)[y_i]
            resp = jnp.exp(log_1m_lam + worst_logp[y_i] + robust_nll)

            return robust_nll, {
                "logits": clean_logits,
                "clean_nll": clean_nll,
                "worst_nll": worst_nll,
                "nll": robust_nll,
                "resp": resp
            }

        grad_fn = jax.grad(per_example, argnums=0, has_aux=True)
        per_example_grads, aux = jax.vmap(
            grad_fn, in_axes=(None, 0, 0)
        )(sampled, x, y)


        batch_logits = aux["logits"]
        nll = jnp.mean(aux["nll"])
        clean_nll = jnp.mean(aux["clean_nll"])
        worst_nll = jnp.mean(aux["worst_nll"])
        if robust_train:
            resp = jnp.mean(aux['resp'])
        else:
            resp = None
            
        for (gw, gb), layer in zip(per_example_grads, layers):
            ghat_w = jnp.mean(gw, axis=0)
            Ghat_w = jnp.mean(gw**2, axis=0)

            ghat_b = jnp.mean(gb, axis=0)
            Ghat_b = jnp.mean(gb**2, axis=0)

            prec = 1.0 / (layer.prior.sigma**2)
            N = layer.n_eff

            # Gradient including the Gaussian-prior contribution.
            grad_w = (
                ghat_w
                + (prec / N) * (layer.mu_w[...] - layer.prior.mu)
            )
            grad_b = (
                ghat_b
                + (prec / N) * (layer.mu_b[...] - layer.prior.mu)
            )

            # Raw EMA states.
            m_w_new = beta1 * layer.m_w[...] + (1.0 - beta1) * grad_w
            m_b_new = beta1 * layer.m_b[...] + (1.0 - beta1) * grad_b

            s_w_new = beta2 * layer.s_w[...] + (1.0 - beta2) * Ghat_w
            s_b_new = beta2 * layer.s_b[...] + (1.0 - beta2) * Ghat_b

            # Stateful bias-correction masses. These survive checkpoint reloads.
            m_ema_weight_new = (
                beta1 * layer.m_ema_weight[...] + (1.0 - beta1)
            )
            s_ema_weight_new = (
                beta2 * layer.s_ema_weight[...] + (1.0 - beta2)
            )

            # Corrected moments used by both optimization and posterior sigma.
            m_w_hat = m_w_new / m_ema_weight_new
            m_b_hat = m_b_new / m_ema_weight_new

            s_w_hat = s_w_new / s_ema_weight_new
            s_b_hat = s_b_new / s_ema_weight_new

            layer.mu_w[...] = layer.mu_w[...] - lr * m_w_hat / (
                s_w_hat + prec / N + eps
            )
            layer.mu_b[...] = layer.mu_b[...] - lr * m_b_hat / (
                s_b_hat + prec / N + eps
            )

            # Save raw EMA states.
            layer.m_w[...] = m_w_new
            layer.m_b[...] = m_b_new
            layer.s_w[...] = s_w_new
            layer.s_b[...] = s_b_new

            # Save correction state.
            layer.m_ema_weight[...] = m_ema_weight_new
            layer.s_ema_weight[...] = s_ema_weight_new
            
        metrics.update(loss=nll, logits=batch_logits, labels=y)

        return {
            "logits": batch_logits,
            "nll": nll,
            "clean_nll": clean_nll,
            "worst_nll": worst_nll,
            "resp": resp
        }

    return vogn_train_step

