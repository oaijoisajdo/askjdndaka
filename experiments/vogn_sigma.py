

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import flax.nnx as nnx
from models import VI_BNN, VOGNLinear, PriorConfig
from training import train, DataConfig, make_dataloaders, make_batch


def layer_curvature_report(layer) -> dict:
    """Per-layer curvature-vs-prior diagnostic for a VOGNLinear."""
    prec = 1.0 / (layer.prior.sigma ** 2)
    n_eff = float(layer.n_eff)

    correction = float(layer.s_ema_weight[...])
    if correction <= 0.0:
        s_w_hat = jnp.zeros_like(layer.s_w[...])
        s_b_hat = jnp.zeros_like(layer.s_b[...])
    else:
        s_w_hat = layer.s_w[...] / correction
        s_b_hat = layer.s_b[...] / correction

    sigma_w, sigma_b = layer.sigma()
    sw = np.asarray(sigma_w).ravel()
    ratio_w = np.asarray(n_eff * s_w_hat / prec).ravel()

    return {
        "n_eff": n_eff,
        "prec": float(prec),
        "s_ema_weight": correction,
        # curvature-to-prior ratio: THE number to watch
        "ratio_median": float(np.median(ratio_w)),
        "ratio_mean": float(np.mean(ratio_w)),
        "ratio_p90": float(np.percentile(ratio_w, 90)),
        "frac_ratio_gt_1": float(np.mean(ratio_w > 1.0)),
        # fraction of weights with literally zero accumulated curvature
        "frac_s_zero": float(np.mean(np.asarray(s_w_hat).ravel() <= 0.0)),
        "sigma_w_median": float(np.median(sw)),
        "sigma_w_p10": float(np.percentile(sw, 10)),
        "sigma_w_max": float(np.max(sw)),
        "sigma_b_median": float(np.median(np.asarray(sigma_b).ravel())),
        # sanity: sigma must never exceed the prior
        "sigma_exceeds_prior": bool(np.max(sw) > layer.prior.sigma * (1 + 1e-6)),
        "mu_w_absmedian": float(np.median(np.abs(np.asarray(layer.mu_w[...]).ravel()))),
    }


def model_curvature_report(model) -> list[dict]:
    layers = list(model.layers) + [model.output_layer]
    out = []
    for i, layer in enumerate(layers):
        rep = layer_curvature_report(layer)
        rep["layer"] = i
        rep["shape"] = list(np.asarray(layer.mu_w[...]).shape)
        out.append(rep)
    return out


def print_report(temperature: float, report: list[dict], clean_acc: float | None = None) -> None:
    head = f"temperature={temperature:<8g}"
    if clean_acc is not None:
        head += f"  clean_acc={clean_acc:.4f}"
    print(head)
    print(
        f"  {'L':<3}{'shape':<14}{'N_eff':>10}"
        f"{'ratio_med':>11}{'ratio_p90':>11}{'%>1':>7}"
        f"{'sig_med':>9}{'sig_max':>9}{'|mu|_med':>10}"
    )
    for r in report:
        print(
            f"  {r['layer']:<3}{str(r['shape']):<14}{r['n_eff']:>10.2e}"
            f"{r['ratio_median']:>11.4f}{r['ratio_p90']:>11.4f}"
            f"{100 * r['frac_ratio_gt_1']:>6.1f}%"
            f"{r['sigma_w_median']:>9.4f}{r['sigma_w_max']:>9.4f}"
            f"{r['mu_w_absmedian']:>10.2e}"
        )
        if r["sigma_exceeds_prior"]:
            print("      !! sigma exceeds prior sigma -- check n_eff/prec wiring")
    print()

def make_model(temperature, seed, n_train):
    prior = PriorConfig(name="gaussian", sigma=0.04)
    layer_kwargs = {
        "dataset_size": n_train,
        "s_init": 0.0,
        "init": "xavier",
        "temperature": temperature,
    }
    return VI_BNN(
        width=512,                       # not 128 -- see below
        depth=1,
        rngs=nnx.Rngs(params=seed, bayes=seed),
        prior=prior,
        layer_cls=VOGNLinear,
        layer_kwargs=layer_kwargs,       # passed as a dict, NOT spread
    )
            

def run_probe(
    *,
    make_model,          # callable: (temperature, seed) -> fresh VI_BNN with VOGNLinear
    train_fn=train,            # your train(); must accept vogn_beta2 / vogn_eps
    temperatures=(0.01, 0.001),
    seed: int = 0,
    lr: float = 1e-3,
    train_steps: int = 3000,
    eval_every: int = 500,
    vogn_beta2: float = 0.99,   # 100-step EMA horizon; 0.999 is too slow to read at 3k steps
    vogn_eps: float = 4e-4,     # damping -- holds effective lr fixed across the sweep
    out_path: str | Path = "vogn_probe.json",
) -> dict:

    train_loader, val_loader, _ = make_dataloaders(DataConfig())
    n_train = len(train_loader.dataset)
    results = {}

    for temp in temperatures:
        model = make_model(temperature=temp, seed=seed, n_train = n_train)

        # No checkpoint reuse: temperature may not be in the checkpoint config,
        # in which case every temperature collides on one path.
        model = train_fn(
            model,
            train_loader,
            val_loader=val_loader,
            save_dir=None,
            seed=seed,
            lr=lr,
            train_steps=train_steps,
            eval_every=eval_every,
            robust_train=False,
            vogn_beta2=vogn_beta2,
            vogn_eps=vogn_eps,
        )

        report = model_curvature_report(model)

        clean_acc = None
        if val_loader is not None:
            correct = total = 0
            for images, labels in val_loader:
                batch = make_batch(images, labels)
                logits = model(batch["image"], sample=False, return_logits=True)
                correct += int(np.sum(np.asarray(jnp.argmax(logits, -1)) == np.asarray(batch["label"])))
                total += len(batch["label"])
            clean_acc = correct / max(total, 1)

        print_report(temp, report, clean_acc)
        results[str(temp)] = {"clean_acc": clean_acc, "layers": report}

    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")
    return results


def verdict(results: dict) -> None:
    """Stage-1 pass/fail against the criteria in the probe design."""
    print("=" * 72)
    for temp, res in results.items():
        last = res["layers"][-1]
        first = res["layers"][0]
        ok_out = last["ratio_median"] > 1.0
        ok_in = first["ratio_median"] > 0.05          # 10x off the observed 0.005 floor
        acc = res["clean_acc"]
        ok_acc = acc is None or acc > 0.90
        tag = "PASS" if (ok_out and ok_in and ok_acc) else "----"
        print(
            f"{tag}  temp={temp:<8} out_layer_ratio={last['ratio_median']:.3f} "
            f"in_layer_ratio={first['ratio_median']:.4f} "
            f"acc={'n/a' if acc is None else f'{acc:.4f}'}"
        )
    print(
        "\nPASS = curvature-driven output layer, input layer moved off the floor,\n"
        "       and clean accuracy did not collapse. If nothing passes but accuracy\n"
        "       held, push temperature lower. If accuracy collapsed, raise vogn_eps."
    )

def main():
    results = run_probe(make_model=make_model)
    verdict(results)
if __name__ == "__main__":
    main()
