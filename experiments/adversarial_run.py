"""
Standard vs robustly-trained models across posterior families.

Orchestration only: everything measured lives in experiments/{attack_eval,
uncertainty, cert_probe, posterior_stats, reporting}.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import flax.nnx as nnx

from models import PriorConfig, build_model
from training import train, load_or_train, DataConfig, make_dataloaders
from certification import Posterior, make_pgd_candidates

from experiments.config import CONFIG
from experiments.posterior_stats import posterior_sigma_stats
from experiments.reporting import full_report, compare_reports
from experiments.utils import value_tag, write_json


# ---------------------------------------------------------------------------
# Data / model helpers
# ---------------------------------------------------------------------------

def flatten_split(loader) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for images, labels in loader:
        images = np.asarray(images)
        xs.append(images.reshape(images.shape[0], -1))
        ys.append(np.asarray(labels))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int32)


def balanced_subset_indices(y, n_per_class) -> np.ndarray:
    """
    Indices of a class-balanced certification subset of the eval split.

    Returning indices rather than the arrays lets every downstream stage
    address the subset by position, so the certification inputs, their
    adversarial counterparts and their per-input records stay aligned by
    construction. Order is the concatenation of per-class blocks; it is fixed
    for a given (y, n_per_class), which is what makes the JSON records
    joinable across seeds and families.
    """
    return np.concatenate([
        np.flatnonzero(y == label)[:n_per_class]
        for label in range(CONFIG.n_classes)
    ])


def build_family_model(family, n_train, seed):
    m = CONFIG.model
    return build_model(
        family=family,
        width=m.width,
        depth=m.depth,
        rngs=nnx.Rngs(params=seed, bayes=seed),
        prior=PriorConfig(name="gaussian", sigma=m.prior_sigma),
        n_train=n_train,
        rho_init=m.rho_init,
        s_init=m.s_init,
        p_drop=m.p_drop,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--pgd", type=int, default=1)
    ap.add_argument("--cert", type=int, default=1)
    ap.add_argument(
        "--eval-split", choices=["val", "test"], default=None,
        help="Split all reported metrics are computed on. Defaults to "
             "CONFIG.eval_split ('val'). Use 'test' only for final numbers.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = CONFIG
    arch = cfg.model.arch_tag

    out_root = Path(args.out_dir).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_dir).expanduser().resolve()
    clean_dir = checkpoint_root / arch / "clean_runs"
    adv_dir = checkpoint_root / arch / "adversarial_runs"
    for d in (out_root, clean_dir, adv_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_dataloaders(DataConfig())

    # Training still watches val_loader (printout only -- `best/` is never
    # restored). Reported numbers come from whichever split eval_split names.
    split = args.eval_split or cfg.eval_split
    eval_loader = {"val": val_loader, "test": test_loader}.get(split)
    if eval_loader is None:
        raise ValueError(f"No usable eval loader for split {split!r}.")

    x_eval, y_eval = flatten_split(eval_loader)
    cert_idx = balanced_subset_indices(y_eval, cfg.cert.per_class)
    n_train = len(train_loader.dataset)
    print(f"Evaluating on the {split} split: "
          f"{len(x_eval)} inputs, {len(cert_idx)} certification inputs")

    eval_kwargs = dict(
        x_eval=x_eval, y_eval=y_eval,
        cert_idx=cert_idx,
        pgd=bool(args.pgd), cert=bool(args.cert),
        # Built ONCE: passed as a static arg into the fused phi2 kernel, so a
        # single instance means a single compilation per phi2 signature.
        pgd_candidates=make_pgd_candidates(
            steps=cfg.cert.candidate_pgd_steps,
            restarts=cfg.cert.candidate_pgd_restarts,
        ),
    )

    for family in cfg.model.families:
        # Checkpoint layout is family-scoped on BOTH branches:
        #   clean_runs/<family>/... and adversarial_runs/<family>/...
        for seed in cfg.seeds[:args.n_seeds]:
            print(f"\n[{family}, seed={seed}] Loading standard model")
            standard_model = load_or_train(
                build_family_model(family, n_train, seed),
                train_loader, train,
                val_loader=val_loader,
                ckpt_dir=clean_dir / family,
                seed=seed,
                **cfg.standard_train.kwargs(),
            )
            standard = full_report(standard_model, seed=seed, **eval_kwargs)
            standard_sigma = posterior_sigma_stats(Posterior(standard_model))

            for point in cfg.robust_sweep:
                print(f"[{family}, seed={seed}] Robust training: "
                      f"eps={point.epsilon}, lambda={point.rob_lam}")
                robust_model = load_or_train(
                    build_family_model(family, n_train, seed),
                    train_loader, train,
                    val_loader=val_loader,
                    ckpt_dir=adv_dir / family,
                    seed=seed,
                    epsilon=point.epsilon,
                    rob_lam=point.rob_lam,
                    **cfg.robust_train.kwargs(),
                )
                robust = full_report(robust_model, seed=seed, **eval_kwargs)

                payload = {
                    "experiment": {
                        "dataset": cfg.dataset,
                        "architecture": arch,
                        "family": family,
                        "seed": seed,
                        "eval_split": split,
                        "eval_inputs": len(x_eval),
                        "certification_inputs": len(cert_idx),
                        "certification_indices": cert_idx.tolist(),
                        "mc_samples": cfg.mc_samples,
                        "pgd_enabled": int(bool(args.pgd)),
                        "certification_enabled": int(bool(args.cert)),
                    },
                    "configuration": {
                        **cfg.as_dict(),
                        "robust_point": {
                            "epsilon": point.epsilon,
                            "rob_lam": point.rob_lam,
                        },
                    },
                    "posterior_sigma": {
                        "standard": standard_sigma,
                        "robust": posterior_sigma_stats(Posterior(robust_model)),
                    },
                    "standard": standard,
                    "robust": robust,
                    "comparison": compare_reports(standard, robust),
                }

                output_path = out_root / arch / family / (
                    f"seed{seed}"
                    f"_eps{value_tag(point.epsilon)}"
                    f"_lam{value_tag(point.rob_lam)}.json"
                )
                write_json(output_path, payload)
                print(f"[{family}, seed={seed}] Saved {output_path.name}; "
                      f"clean Δ={payload['comparison']['clean_accuracy_delta']:+.4f}")


if __name__ == "__main__":
    main()
