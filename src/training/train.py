from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from .data_loader import make_batch
from .losses import (
    make_mc_dropout_loss,
    make_bbb_loss,
    make_eval_loss,
)
from .checkpointing import save_model
from .vogn_train import make_vogn_train_step

ModelType = Literal["deterministic", "mc_dropout", "bbb", "vogn"]  # CHANGED


def make_loss_fn(
    *,
    model_type: ModelType,
    n_train: int,
    beta: float,
    robust_train: bool = False,
):
    if model_type == "bbb":
        return make_bbb_loss(
            n_train=n_train,
            beta=beta,
            robust_train=robust_train,
        )
 
    if model_type in ("mc_dropout", "deterministic"):
        return make_mc_dropout_loss(robust_train=robust_train)
    
 
def make_train_step(loss_fn: Callable):
    @nnx.jit
    def train_step(
        model,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        batch,
        rngs: nnx.Rngs,
        epsilon=0.0,   
        rob_lam=1.0,
    ):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(model, batch, rngs, epsilon, rob_lam)  # CHANGED
 
        optimizer.update(model, grads)
 
        metrics.update(
            loss=loss,
            logits=aux["logits"],
            labels=batch["label"],
        )
 
        return aux
 
    return train_step

def make_eval_step(eval_loss_fn: Callable):
    @nnx.jit
    def eval_step(
        model,
        metrics: nnx.MultiMetric,
        batch,
    ):
        loss, aux = eval_loss_fn(model, batch, None)

        metrics.update(
            loss=loss,
            logits=aux["logits"],
            labels=batch["label"],
        )

        return aux

    return eval_step


def evaluate(
    model,
    loader,
    eval_step: Callable,
) -> dict[str, float]:
    metrics = make_metrics()

    for images, labels in loader:
        batch = make_batch(images, labels)
        eval_step(model, metrics, batch)

    return compute_metrics(metrics)


def train(
    model,
    train_loader,
    *,
    val_loader=None,
    save_dir: str | Path | None = None,
    seed: int = 0,
    lr: float = 1e-3,
    train_steps: int = 500,
    eval_every: int = 100,
    beta: float = 1.0,
    robust_train: bool = False,
    epsilon: float = 0.0,
    epsilon_warmup_steps: int = 0, 
    rob_lam: float = 1.0,
    save_best: bool = False,
    vogn_beta1: float = 0.9, 
    vogn_beta2: float = 0.999,  
    vogn_eps: float = 1e-8,    
):
    model_type = model.model_type

    n_train = len(train_loader.dataset)

    if model_type == "vogn":
        # Sanity check: VOGNLinear needs N at construction time 
        assert model.output_layer.dataset_size == n_train, (
            "VOGN layers were built with a different dataset_size than "
            f"this train_loader implies ({model.output_layer.dataset_size} "
            f"!= {n_train})."
        )
        optimizer = None #  no optax optimizer for vogn
        step_fn = make_vogn_train_step(  # CHANGED
            lr=lr, beta1=vogn_beta1, beta2=vogn_beta2, eps=vogn_eps,
            robust_train=robust_train,
        )


    else:
        optimizer = make_optimizer(model, lr)

        loss_fn = make_loss_fn(
            model_type=model_type,
            n_train=n_train,
            beta=beta,
            robust_train=robust_train,
        )
        step_fn = make_train_step(loss_fn)

    train_metrics = make_metrics()
    rngs = make_rngs(model_type, seed)

    eval_step = make_eval_step(make_eval_loss(model_type))

    best_val_loss = float("inf")

    save_dir = Path(save_dir).resolve() if save_dir is not None else None

    step = 0
    rob_lam_t = rob_lam
    print(f"eps scheduled from 0 to {epsilon} over {epsilon_warmup_steps} steps")
    print(f"lamda fixed at {rob_lam}")



    while step < train_steps:
        for images, labels in train_loader:
            batch = make_batch(images, labels)

            step += 1

            if robust_train:
                eps_t = linear_schedule(
                    step,
                    start_step=0,
                    end_step=epsilon_warmup_steps,
                    start_val=0.0,
                    end_val=epsilon,
                )

                #rob_lam_t=linear_schedule(step, start_step=0, end_step=epsilon_warmup_steps, start_val=1.0, end_val = rob_lam)
                rob_lam_t = rob_lam
            else:
                eps_t = 0.0
                rob_lam_t = 1.0
            if model_type == "vogn":

                aux = step_fn(  
                    model, train_metrics, batch, rngs, step,
                    epsilon=eps_t, rob_lam=rob_lam_t,
                )
            else:
                aux = step_fn( 
                    model, optimizer, train_metrics, batch, rngs,
                    epsilon=eps_t, rob_lam=rob_lam_t,
                )

            if step % eval_every == 0 or step == train_steps:
                train_res = compute_metrics(train_metrics)
                train_metrics.reset()
                msg = (
                    f"step={step:4d} "
                    f"train_acc={train_res['accuracy']:.4f}"
                )

                if val_loader is not None:
                    val_res = evaluate(model, val_loader, eval_step)
                    msg += (
                        f" val_acc={val_res['accuracy']:.4f}"
                    )

                    if save_dir is not None and save_best and val_res["loss"] < best_val_loss:
                        best_val_loss = val_res["loss"]
                        maybe_save(model, save_dir / "best")

                if model_type == "bbb":
                    msg += (
                        f" kl={float(aux['kl']):.2f}"
                        f" kl_scaled={float(aux['kl_scaled']):.4f}"
                        f" nll={float(aux['nll']):.4f}"
                    )

                if model_type == "vogn": 
                    msg += (
                        f" nll={float(aux['nll']):.4f}"
                        
                    )
                if robust_train:
                    msg += (
                        f" eps={float(eps_t):.4f}" 
                        f" clean_nll={float(aux['clean_nll']):.4f}"
                        f" worst_nll={float(aux['worst_nll']):.4f}"
                        f" resp={float(aux['resp']):.4f}"
                    )

                print(msg)

            if step >= train_steps:
                break

    if save_dir is not None:
        maybe_save(model, save_dir)

    return model

def linear_schedule(step, start_step, end_step, start_val, end_val):
    """Linear ramp from `start_val` to `end_val` between `start_step` and
    `end_step`, held constant outside that range. Pass the result straight
    into `vogn_train_step(..., epsilon=..., rob_lam=...)` -- since epsilon
    and rob_lam are traced arguments (not closure constants), varying this
    value across steps does not trigger recompilation.
 
    Typical use (Gowal et al. 2018-style epsilon warm-up):
        eps_t = linear_schedule(step, start_step=0, end_step=2000,
                                 start_val=0.0, end_val=0.3)
    """
    step = jnp.asarray(step, dtype=jnp.float32)
    denom = jnp.maximum(jnp.asarray(end_step - start_step, dtype=jnp.float32), 1.0)
    frac = jnp.clip((step - start_step) / denom, 0.0, 1.0)
    return start_val + frac * (end_val - start_val)

def make_optimizer(model, lr: float):
    return nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)


def make_metrics():
    return nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )


def make_rngs(model_type: ModelType, seed: int):
    if model_type == "mc_dropout":
        return nnx.Rngs(dropout=seed)

    if model_type in ("bbb", "vogn"): 
        return nnx.Rngs(bayes=seed)

    return nnx.Rngs(seed)


def compute_metrics(metrics: nnx.MultiMetric) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.compute().items()}


def maybe_save(model, path):
    if path is None:
        return
    save_model(model, path)