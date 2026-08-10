import argparse
from pathlib import Path
import flax.nnx as nnx
from experiments.config import CONFIG
from models import PriorConfig, build_model
from training import train, load_or_train, DataConfig, make_dataloaders


def build_family_model(family, n_train, seed):
    m = CONFIG.model
    prior = PriorConfig(name="gaussian", sigma=m.prior_sigma)
    return build_model(
        family=family,
        width=m.width,
        depth=m.depth,
        rngs=nnx.Rngs(params=seed, bayes=seed),
        prior=prior,
        n_train=n_train,
        rho_init=m.rho_init,
        s_init=m.s_init,
        p_drop=m.p_drop,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--n-seeds", type=int, default=5)

    args = ap.parse_args()

    cfg = CONFIG
    arch = cfg.model.arch_tag

    out_root = Path(args.out_dir).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_dir).expanduser().resolve()
    clean_dir = checkpoint_root / arch / "clean_runs"
    adv_dir = checkpoint_root / arch / "adversarial_runs"
    for d in (out_root, clean_dir, adv_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = make_dataloaders(DataConfig())

    n_train = len(train_loader.dataset)

    for family in cfg.model.families:
        clean_family_dir = clean_dir / family
        family_checkpoint_dir = adv_dir / family

        for seed in cfg.seeds[:args.n_seeds]:
            print(f"\n[{family}, seed={seed}] Loading standard model")
            standard_model = build_family_model(family, n_train, seed)
            standard_model = load_or_train(
                standard_model, train_loader, train,
                val_loader=val_loader,
                ckpt_dir=clean_family_dir,
                seed=seed,
                **cfg.standard_train.kwargs(),
            )

            for point in cfg.robust_sweep:
                print(
                    f"[{family}, seed={seed}] Robust training: "
                    f"eps={point.epsilon}, lambda={point.rob_lam}"
                )
                robust_model = build_family_model(family, n_train, seed)
                robust_model = load_or_train(
                    robust_model, train_loader, train,
                    val_loader=val_loader,
                    ckpt_dir=family_checkpoint_dir,
                    seed=seed,
                    epsilon=point.epsilon,
                    rob_lam=point.rob_lam,
                    **cfg.robust_train.kwargs(),
                )


if __name__ == "__main__":
    main()