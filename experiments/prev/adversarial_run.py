
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from models import PriorConfig, build_model
from training import train, load_or_train, DataConfig, make_dataloaders
from attacks import (
    pgd_attack,
    mc_dropout_logits_fn,
    bbb_logits_fn,
)
from certification import (
    Posterior,
    estimate_probabilistic_robustness_bounds,
    make_phi2,
    propagate_deterministic_LBP,
    make_pgd_candidates,
)

from experiments.config import CONFIG
from experiments.predictive_utils import mc_probs_batched, clean_report
from experiments.utils import epsilon_key, value_tag, write_json

from certification.bound_propagation import forward_layers


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def balanced_subset(x, y, n_per_class):
    indices = np.concatenate([
        np.flatnonzero(y == label)[:n_per_class]
        for label in range(10)
    ])
    return x[indices], y[indices]


def _split_arrays(loader) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for images, labels in loader:
        images = np.asarray(images)
        xs.append(images.reshape(images.shape[0], -1))
        ys.append(np.asarray(labels))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int32)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

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


def make_logits_fn(model):
    if model.model_type in {"mc_dropout", "deterministic"}:
        return mc_dropout_logits_fn(model)
    if model.model_type in ("bbb", "vogn"):
        return bbb_logits_fn(model)
    raise ValueError(f"Unknown model_type: {model.model_type}")


def _is_gaussian_posterior(posterior) -> bool:
    return hasattr(posterior, "mu_w") and hasattr(posterior, "sigma_w")


def _is_dropout_posterior(posterior) -> bool:
    return hasattr(posterior, "W") and hasattr(posterior, "b")


def posterior_sigma_stats(posterior) -> dict:
    """
    Per-layer posterior spread summaries -- Phase E data, costs nothing.

    Polymorphic over the posterior families exported by `certification`:

      * Gaussian (BBB / VOGN): sigma is explicit.
      * Bernoulli (MC-dropout): there is no sigma, but inverted dropout
        induces a well-defined weight-space spread. Row i of W[k+1] is
        multiplied by m_i / (1 - p) with m_i ~ Bernoulli(1 - p), so that
        entry has mean w and standard deviation |w| * sqrt(p / (1 - p)).
        Reporting that puts dropout on the same axis as the Gaussian
        families, which is what the sigma-confound comparison needs.

    Returns a dict with a `kind` tag so downstream analysis never has to
    guess which convention a payload used.
    """
    if _is_gaussian_posterior(posterior):
        layers = []
        for i in range(len(posterior.mu_w)):
            sw = np.asarray(posterior.sigma_w[i])
            sb = np.asarray(posterior.sigma_b[i])
            mw = np.asarray(posterior.mu_w[i])
            layers.append({
                "layer": i,
                "sigma_w_mean": float(sw.mean()),
                "sigma_w_median": float(np.median(sw)),
                "sigma_w_max": float(sw.max()),
                "sigma_b_mean": float(sb.mean()),
                # Scale-free: sigma relative to the weight magnitude it sits
                # on. Comparable across families and across layers.
                "relative_spread_mean": float(
                    (sw / (np.abs(mw) + 1e-12)).mean()
                ),
            })
        return {"kind": "gaussian", "layers": layers}

    if _is_dropout_posterior(posterior):
        p = float(posterior.p)
        # sd of (m / (1 - p)) for m ~ Bernoulli(1 - p)
        mask_cv = math.sqrt(p / (1.0 - p)) if p > 0.0 else 0.0
        layers = []
        for k in range(len(posterior.W)):
            w = np.asarray(posterior.W[k])
            # Mask k acts on the rows of W[k + 1]; W[0] is deterministic.
            stochastic = k > 0
            sw = np.abs(w) * mask_cv if stochastic else np.zeros_like(w)
            layers.append({
                "layer": k,
                "stochastic": bool(stochastic),
                "sigma_w_mean": float(sw.mean()),
                "sigma_w_median": float(np.median(sw)),
                "sigma_w_max": float(sw.max()),
                "sigma_b_mean": 0.0,          # biases are not dropped
                "relative_spread_mean": float(mask_cv if stochastic else 0.0),
            })
        return {
            "kind": "bernoulli_dropout",
            "p_drop": p,
            "mask_cv": mask_cv,
            "layers": layers,
        }

    return {"kind": "unknown", "layers": None}


# ---------------------------------------------------------------------------
# Per-input uncertainty (clean inputs, ensemble predictor)
# ---------------------------------------------------------------------------

def _ensemble_probs(probs, n_inputs: int) -> np.ndarray:
    """Convert probabilities of shape (S, N, C) to (N, C)."""
    p = np.asarray(probs)

    if p.ndim != 3:
        raise ValueError(f"Expected probabilities with shape (S, N, C), got {p.shape}.")

    if p.shape[1] != n_inputs:
        raise ValueError(
            f"Expected {n_inputs} inputs on axis 1, got shape {p.shape}."
        )

    return p.mean(axis=0)


def mc_probs_from_posterior(posterior, x, *, n_samples: int, key) -> np.ndarray:
    """
    Per-sample predictive probabilities, shape (S, N, C), drawn from the
    posterior object itself.

    Both Gaussian and Bernoulli posteriors duck-type `sample(key) ->
    [(w, b), ...]`, and `forward_layers` batches over the leading input
    axis, so this works for every family without touching model internals.

    Using the posterior here (rather than the model's own MC path) means the
    uncertainty measures and the certified P_safe values in the same payload
    are computed from an identical representation of the weight distribution
    -- which is exactly what the Phase C alignment analysis compares.

    For a deterministic model every draw coincides, so the mutual
    information is 0 by construction; that is the correct answer, not a bug.
    """
    xs = jnp.asarray(x)
    keys = jax.random.split(key, n_samples)
    samples = []
    for k in keys:
        layers = posterior.sample(k)
        logits = forward_layers(layers, xs)
        samples.append(jax.nn.softmax(jnp.asarray(logits), axis=-1))
    return np.asarray(jnp.stack(samples))          # (S, N, C)


def per_input_uncertainty(probs_snc: np.ndarray, y) -> dict[str, list]:
    """
    Entropy, mutual information, confidence and correctness per input.

    Expects probs of shape (S, N, C) exactly as produced by
    mc_probs_from_posterior -- no axis guessing.

        H[E_w p]                        total predictive uncertainty
        H[E_w p] - E_w H[p]             mutual information (epistemic)
    """
    p = np.asarray(probs_snc)
    if p.ndim != 3:
        raise ValueError(f"Expected (S, N, C) probabilities; got {p.shape}.")
    eps = 1e-12
    mean_p = p.mean(axis=0)                             # (N, C)
    entropy = -(mean_p * np.log(mean_p + eps)).sum(-1)
    sample_entropy = -(p * np.log(p + eps)).sum(-1)     # (S, N)
    mutual_information = entropy - sample_entropy.mean(axis=0)
    prediction = mean_p.argmax(-1)
    return {
        "n_mc_samples": int(p.shape[0]),
        "predictive_entropy": entropy.tolist(),
        "mutual_information": mutual_information.tolist(),
        "expected_entropy": sample_entropy.mean(axis=0).tolist(),
        "confidence": mean_p.max(-1).tolist(),
        "correct": (prediction == np.asarray(y)).astype(int).tolist(),
    }


# ---------------------------------------------------------------------------
# Evaluation: clean + PGD (validation set, aggregate)
# ---------------------------------------------------------------------------

def evaluate(model, x, y, seed):
    probs = mc_probs_batched(
        model, x,
        n_samples=CONFIG.mc_samples,
        seed=seed,
        model_type=model.model_type,
    )
    return clean_report(probs, y)


def evaluate_pgd(model, x, y, eps, seed, *, return_x=False):
    """Attack with one seed stream, evaluate with a disjoint one."""
    a = CONFIG.attack

    x_adv = pgd_attack(
        make_logits_fn(model),
        x,
        y,
        eps=eps,
        key=jax.random.PRNGKey(seed),
        stochastic=True,
        mc_samples=CONFIG.mc_samples,
        mode=a.mode,
        targeted=False,
        steps=a.steps,
        random_start=a.random_start,
    )

    if return_x:
        return x_adv

    return evaluate(
        model,
        x_adv,
        y,
        seed + a.eval_seed_offset,
    )


def pgd_hits_per_input(model, x, y, eps, seed) -> list[int]:
    """1 if the ensemble predictor is broken at this input under PGD."""
    a = CONFIG.attack
    x_adv = pgd_attack(
        make_logits_fn(model),
        x, y,
        eps=eps,
        key=jax.random.PRNGKey(seed + 1),   # distinct from val-set attack
        stochastic=True,
        mc_samples=CONFIG.mc_samples,
        mode=a.mode,
        targeted=False,
        steps=a.steps,
        random_start=a.random_start,
    )
    probs = mc_probs_batched(
        model, x_adv,
        n_samples=CONFIG.mc_samples,
        seed=seed + a.eval_seed_offset + 1,
        model_type=model.model_type,
    )
    p = _ensemble_probs(probs, len(y))
    return (p.argmax(-1) != np.asarray(y)).astype(int).tolist()


# ---------------------------------------------------------------------------
# Certification probe
# ---------------------------------------------------------------------------

def _mean_network_layers(posterior):
    """
    The zero-spread limit of the posterior: one deterministic network.

      * Gaussian (BBB / VOGN): the posterior mean, (mu_w, mu_b).
      * Bernoulli (MC-dropout): dropout disabled. Under inverted dropout the
        mask contributes m / (1 - p) with mean exactly 1, so the mean network
        is the raw (W, b) with NO 1/(1 - p) rescaling -- i.e. the ordinary
        test-time forward pass. Do not build this from `layers_for_masks`
        with an all-ones mask; that would scale every row by 1/(1 - p).

    Returns None when the posterior exposes neither convention, in which case
    the sigma-confound control is skipped rather than crashing the run.
    """
    if _is_gaussian_posterior(posterior):
        return [
            (jnp.asarray(posterior.mu_w[i]), jnp.asarray(posterior.mu_b[i]))
            for i in range(len(posterior.mu_w))
        ]

    if _is_dropout_posterior(posterior):
        return [
            (jnp.asarray(posterior.W[k]), jnp.asarray(posterior.b[k]))
            for k in range(len(posterior.W))
        ]

    return None


def cert_probe(model, x, y, *, seed, pgd_candidates) -> dict:
    """
    Statistical P_safe probe per input per eps, plus the sigma -> 0 control.

    Serializes per-input arrays: the aggregates alone cannot feed the Phase C
    rank-correlation / ROC analysis, and the delta tables in v1 were censored
    by the theta floor/ceiling.
    """
    c = CONFIG.cert
    posterior = Posterior(model)
    mean_layers = _mean_network_layers(posterior)
    base_key = jax.random.key(seed)

    results = {}
    for eps in c.eps_values:
        per_input = {
            "true_label": [int(v) for v in y],
            # theta-shifted Massart bounds (guarantees; censored at 0/0.925)
            "p_safe_lower": [],
            "p_safe_upper": [],
            # un-censored point estimates (use these for ranking/correlation)
            "p_safe_point": [],          # 1 - p_hat, unknown counted unsafe
            "p_safe_point_optimistic": [],  # 1 - p_hat, unknown counted safe
            "p_safe_ci_low": [],         # Clopper-Pearson on the pessimistic side
            "p_safe_ci_high": [],
            # verifier diagnostics
            "unknown_frac": [],          # from the pessimistic-side stream
            "n_samples_used": [],
            "n_property_evaluations": [],
            # sigma -> 0 control: verdict for the posterior-mean network
            "mean_net_verdict": [],
        }

        for i in range(len(x)):
            x_star = x[i][None, :]
            x_l = np.clip(x_star - eps, 0.0, 1.0)
            x_u = np.clip(x_star + eps, 0.0, 1.0)

            phi = make_phi2(
                x_star=x_star,
                x_L=x_l,
                x_U=x_u,
                y_dim=10,
                class_selection="true_label",
                true_label=int(y[i]),
                propagate_fn=propagate_deterministic_LBP,
                ibp_first=c.ibp_first,
                candidates_fn=pgd_candidates,
            )

            key = jax.random.fold_in(base_key, i)
            bounds = estimate_probabilistic_robustness_bounds(
                posterior=posterior,
                property_fn=phi,
                key=key,
                theta=c.theta,
                gamma=c.gamma,
                alpha=c.alpha,
            )

            pess = bounds.unknown_as_violation   # unknown counted as unsafe
            opt = bounds.unknown_as_safe         # unknown counted as safe

            per_input["p_safe_lower"].append(float(bounds.robustness_lower))
            per_input["p_safe_upper"].append(float(bounds.robustness_upper))
            per_input["p_safe_point"].append(float(1.0 - pess.p_hat))
            per_input["p_safe_point_optimistic"].append(float(1.0 - opt.p_hat))
            # CP interval on violation prob -> flip to robustness
            per_input["p_safe_ci_low"].append(float(1.0 - pess.ci_high))
            per_input["p_safe_ci_high"].append(float(1.0 - pess.ci_low))
            per_input["unknown_frac"].append(float(pess.unk / pess.n))
            per_input["n_samples_used"].append(int(pess.n))
            per_input["n_property_evaluations"].append(
                int(bounds.n_property_evaluations)
            )
            per_input["mean_net_verdict"].append(
                phi(mean_layers, jax.random.fold_in(key, 999_983))
                if mean_layers is not None else None
            )

        p_lo = np.asarray(per_input["p_safe_lower"])
        p_hi = np.asarray(per_input["p_safe_upper"])
        p_pt = np.asarray(per_input["p_safe_point"])
        mean_verdicts = per_input["mean_net_verdict"]

        results[f"{eps:g}"] = {
            "n_inputs": len(x),
            # Bound-based aggregates (comparable to v1, still censored)
            "mean_p_safe_lower": float(p_lo.mean()),
            "mean_p_safe_upper": float(p_hi.mean()),
            "mean_interval_width": float((p_hi - p_lo).mean()),
            "frac_p_safe_lower_geq_0p5": float(np.mean(p_lo >= 0.5)),
            # Point-estimate aggregates (uncensored; report these for deltas)
            "mean_p_safe_point": float(p_pt.mean()),
            "median_p_safe_point": float(np.median(p_pt)),
            "frac_p_safe_point_geq_0p9": float(np.mean(p_pt >= 0.9)),
            # Verifier diagnostics
            "mean_unknown_frac": float(np.mean(per_input["unknown_frac"])),
            "mean_property_evaluations": float(
                np.mean(per_input["n_property_evaluations"])
            ),
            # Zero-spread control aggregate (None if unavailable)
            "mean_net_frac_safe": (
                float(np.mean([v == "safe" for v in mean_verdicts]))
                if mean_layers is not None else None
            ),
            "mean_net_frac_unsafe": (
                float(np.mean([v == "unsafe" for v in mean_verdicts]))
                if mean_layers is not None else None
            ),
            "per_input": per_input,
        }

    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_reports(standard, robust):
    comparison = {
        "clean_accuracy_delta": float(
            robust["clean"]["accuracy"] - standard["clean"]["accuracy"]
        )
    }

    if "pgd" in standard:
        comparison["pgd_accuracy_delta"] = {
            eps: float(
                robust["pgd"][eps]["accuracy"] - standard["pgd"][eps]["accuracy"]
            )
            for eps in standard["pgd"]
        }

    if "certification" in standard:
        # Bound-based delta kept for continuity, but at eps where the
        # standard model's lower bound is floored at 0 this is NOT a delta --
        # it is the robust value with a censored baseline. Prefer the
        # point-estimate delta below.
        comparison["mean_p_safe_lower_delta"] = {
            eps: float(
                robust["certification"][eps]["mean_p_safe_lower"]
                - standard["certification"][eps]["mean_p_safe_lower"]
            )
            for eps in standard["certification"]
        }
        comparison["mean_p_safe_point_delta"] = {
            eps: float(
                robust["certification"][eps]["mean_p_safe_point"]
                - standard["certification"][eps]["mean_p_safe_point"]
            )
            for eps in standard["certification"]
        }
        # Sigma-confound decomposition: how much of the P_safe delta is the
        # mean network moving vs the posterior contracting around it?
        def _mean_net_delta(eps):
            r = robust["certification"][eps]["mean_net_frac_safe"]
            s = standard["certification"][eps]["mean_net_frac_safe"]
            return None if (r is None or s is None) else float(r - s)

        comparison["mean_net_frac_safe_delta"] = {
            eps: _mean_net_delta(eps) for eps in standard["certification"]
        }

    return comparison


# ---------------------------------------------------------------------------
# Per-model evaluation bundle
# ---------------------------------------------------------------------------

def full_report(model, *, x_eval, y_eval, x_cert, y_cert, seed,
                pgd, cert, pgd_candidates) -> dict:
    
    a = CONFIG.attack
    posterior = Posterior(model)

    report = {
        "clean": evaluate(model, x_eval, y_eval, seed)
    }

    # Common posterior samples for clean and attacked certification inputs.
    # This reduces Monte Carlo noise in U(x_adv) - U(x).
    cert_eval_key = jax.random.key(
        seed + a.eval_seed_offset + 7
    )

    probs_clean_cert = mc_probs_from_posterior(
        posterior,
        x_cert,
        n_samples=CONFIG.mc_samples,
        key=cert_eval_key,
    )
    report["cert_subset_uncertainty"] = per_input_uncertainty(
        probs_clean_cert,
        y_cert,
    )

    if pgd:
        report["pgd"] = {}
        report["cert_subset_pgd_hit"] = {}

        pgd_eps = {
            epsilon_key(eps): eps
            for eps in CONFIG.attack.pgd_eps
        }
        cert_eps = {
            epsilon_key(eps)
            for eps in CONFIG.cert.eps_values
        }

        missing = cert_eps - set(pgd_eps)
        if missing:
            raise ValueError(
                "Certification epsilons missing from the PGD grid: "
                f"{sorted(missing, key=float)}"
            )

        for eps_key, eps in pgd_eps.items():
            # Use a distinct attack stream for the certification subset.
            x_adv_cert = evaluate_pgd(
                model,
                x_cert,
                y_cert,
                eps,
                seed + 1,
                return_x=True,
            )

            # Independent of the samples used to construct the attack.
            probs_adv_cert = mc_probs_from_posterior(
                posterior,
                x_adv_cert,
                n_samples=CONFIG.mc_samples,
                key=cert_eval_key,
            )
            adv_uncertainty = per_input_uncertainty(
                probs_adv_cert,
                y_cert,
            )

            # Keep existing aggregate metrics at the top level of the block.
            pgd_report = evaluate_pgd(
                model,
                x_eval,
                y_eval,
                eps,
                seed,
            )
            pgd_report["cert_subset_uncertainty"] = adv_uncertainty
            report["pgd"][eps_key] = pgd_report

            # Keep this legacy field only on certification radii.
            if eps_key in cert_eps:
                adv_correct = np.asarray(
                    adv_uncertainty["correct"],
                    dtype=bool,
                )
                report["cert_subset_pgd_hit"][eps_key] = (
                    ~adv_correct
                ).astype(int).tolist()

    if cert:
        report["certification"] = cert_probe(
            model,
            x_cert,
            y_cert,
            seed=seed,
            pgd_candidates=pgd_candidates,
        )

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
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
    args = ap.parse_args()

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
    if split not in ("val", "test"):
        raise ValueError(f"eval_split must be 'val' or 'test'; got {split!r}.")
    eval_loader = val_loader if split == "val" else test_loader
    if eval_loader is None:
        raise ValueError(f"make_dataloaders returned no {split} loader.")

    x_eval, y_eval = _split_arrays(eval_loader)
    x_cert, y_cert = balanced_subset(x_eval, y_eval, cfg.cert.per_class)
    n_train = len(train_loader.dataset)
    print(f"Evaluating on the {split} split: "
          f"{len(x_eval)} inputs, {len(x_cert)} certification inputs")

    pgd = (args.pgd == 1)
    cert = (args.cert == 1)

    # Built ONCE: passed as a static arg into the fused phi2 kernel, so a
    # single instance means a single compilation per phi2 signature.
    pgd_candidates = make_pgd_candidates(
        steps=cfg.cert.candidate_pgd_steps,
        restarts=cfg.cert.candidate_pgd_restarts,
    )

    eval_kwargs = dict(
        x_eval=x_eval, y_eval=y_eval,
        x_cert=x_cert, y_cert=y_cert,
        pgd=pgd, cert=cert,
        pgd_candidates=pgd_candidates,
    )

    for family in cfg.model.families:
        # Existing checkpoint layout is family-scoped on BOTH branches:
        #   clean_runs/<family>/<family>_standard_<hash>
        #   adversarial_runs/<family>/...
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

            standard = full_report(standard_model, seed=seed, **eval_kwargs)
            standard_sigma = posterior_sigma_stats(Posterior(standard_model))

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

                robust = full_report(robust_model, seed=seed, **eval_kwargs)
                robust_sigma = posterior_sigma_stats(Posterior(robust_model))

                payload = {
                    "experiment": {
                        "dataset": cfg.dataset,
                        "architecture": arch,
                        "family": family,
                        "seed": seed,
                        "eval_split": split,
                        "eval_inputs": len(x_eval),
                        "certification_inputs": len(x_cert),
                        "mc_samples": cfg.mc_samples,
                        "pgd_enabled": int(pgd),
                        "certification_enabled": int(cert),
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
                        "robust": robust_sigma,
                    },
                    "standard": standard,
                    "robust": robust,
                    "comparison": compare_reports(standard, robust),
                }

                filename = (
                    f"seed{seed}"
                    f"_eps{value_tag(point.epsilon)}"
                    f"_lam{value_tag(point.rob_lam)}.json"
                )
                output_path = out_root / arch / family / filename
                write_json(output_path, payload)

                clean_delta = payload["comparison"]["clean_accuracy_delta"]
                print(
                    f"[{family}, seed={seed}] Saved {output_path.name}; "
                    f"clean Δ={clean_delta:+.4f}"
                )


if __name__ == "__main__":
    main()