from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint as ocp

Array = jax.Array

class MC_BNN(nnx.Module):
    def __init__(
        self,
        d_in: int = 784,
        d_out: int = 10,
        width: int = 24,
        depth: int = 1,
        p_drop: float = 0.2,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.width = width
        self.depth = depth
        self.p_drop = p_drop

        self.layers = nnx.List([
            nnx.Linear(d_in if i == 0 else width, width, rngs=rngs)
            for i in range(depth)
        ])
        self.output_layer = nnx.Linear(width, d_out, rngs=rngs)
        self.dropout = nnx.Dropout(rate=p_drop)

    def __call__(
        self,
        x: Array,
        *,
        disable_dropout: bool = False,
        return_logits: bool = False,
        rngs: nnx.Rngs | None = None,
    ):
        for layer in self.layers:
            x = nnx.relu(layer(x))
            x = self.dropout(x, deterministic=disable_dropout, rngs=rngs)

        logits = self.output_layer(x)
        if return_logits:
            return logits
        return nnx.softmax(logits)

    def predict_proba(self, x: Array):
        return self(x, disable_dropout=True)

    def predict_mc(self, x: Array, n_samples: int = 30, *, key: Array):
        keys = jax.random.split(key, n_samples)

        def one_sample(k):
            rngs = nnx.Rngs(dropout=k)
            return self(x, disable_dropout=False, rngs=rngs)

        samples = jax.vmap(one_sample)(keys)
        return samples, samples.mean(axis=0), samples.var(axis=0)

    def clean_and_ibp_logits(
        self,
        x: Array,
        *,
        epsilon: float,
        rngs: nnx.Rngs,
        disable_dropout: bool = False,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        clean = x
        x_l = jnp.clip(x - epsilon, clip_min, clip_max)
        x_u = jnp.clip(x + epsilon, clip_min, clip_max)

        for layer in self.layers:
            w, b = layer.kernel.value, layer.bias.value

            clean = nnx.relu(clean @ w + b)
            x_l, x_u = dense_ibp(x_l, x_u, w, b)
            x_l, x_u = nnx.relu(x_l), nnx.relu(x_u)

            if not disable_dropout and self.p_drop > 0.0:
                keep = jax.random.bernoulli(
                    rngs.dropout(), 1.0 - self.p_drop, clean.shape
                ).astype(clean.dtype)
                m = keep / (1.0 - self.p_drop)   # >= 0, so monotone
                clean, x_l, x_u = clean * m, x_l * m, x_u * m

        w, b = self.output_layer.kernel.value, self.output_layer.bias.value
        clean_logits = clean @ w + b
        logit_l, logit_u = dense_ibp(x_l, x_u, w, b)
        return clean_logits, logit_l, logit_u




    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpointer = ocp.StandardCheckpointer()
        state = nnx.state(self)

        checkpointer.save(path / "state", state)
        checkpointer.save(
            path / "config",
            {
                "d_in": self.d_in,
                "d_out": self.d_out,
                "width": self.width,
                "depth": self.depth,
                "p_drop": self.p_drop,
            },
        )
        checkpointer.wait_until_finished()

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 0):
        path = Path(path)

        with ocp.StandardCheckpointer() as checkpointer:
            cfg = checkpointer.restore(path / "config")

            abstract_model = nnx.eval_shape(lambda: cls(**cfg, rngs=nnx.Rngs(seed)))
            graphdef, abstract_state = nnx.split(abstract_model)

            restored_state = checkpointer.restore(path / "state", abstract_state)

        model = nnx.merge(graphdef, restored_state)
        return model
    
    @property
    def model_type(self) -> str:
        if self.p_drop <= 0.0:
            return "deterministic"
        return "mc_dropout"
    
    def get_config(self) -> dict:
        return {
            "model_type": self.model_type,  # "mc_dropout" or "deterministic"
            "d_in": self.d_in,
            "d_out": self.d_out,
            "width": self.width,
            "depth": self.depth,
            "p_drop": self.p_drop,
        }

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
