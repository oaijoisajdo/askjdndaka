"""src/evaluation/predictive.py

MC predictive distributions and clean-data calibration baselines.

The MC probability stack `probs[S, N, C]` produced here is deliberately kept
and written to disk: Phase C's predictive entropy, mutual information and
expected variance are all cheap functions of exactly this array, and
recomputing it per measure would re-sample the posterior three times and give
three subtly different answers.

    H[y|x]       = -sum_c pbar_c log pbar_c          (total)
    E_w H[y|x,w] = mean_s -sum_c p_sc log p_sc       (aleatoric)
    MI           = H[y|x] - E_w H[y|x,w]             (epistemic)

Convention: `model(x)` returns *softmax probabilities* (VI_BNN passes
return_logits=False by default), so nothing here applies a second softmax.
Confirm MC_BNN does the same -- if it returns logits, add the softmax in
`_forward` rather than in every metric.
"""

from __future__ import annotations

import inspect

import numpy as np
import jax.numpy as jnp
import flax.nnx as nnx


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
# VI_BNN.__call__ takes (x, *, rngs, sample, ...); MC_BNN's signature may
# differ (deterministic=/train=/rngs= are all plausible). Rather than guess,
# pass only the keywords the callee actually accepts.

def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _forward(model, x, *, rngs=None, sample: bool = True):
    kwargs = {}
    if rngs is not None and _accepts(model.__call__, "rngs"):
        kwargs["rngs"] = rngs
    if _accepts(model.__call__, "sample"):
        kwargs["sample"] = sample
    elif _accepts(model.__call__, "deterministic"):
        kwargs["deterministic"] = not sample
    return model(x, **kwargs)


def mc_probs(
    model,
    x: jnp.ndarray,
    *,
    n_samples: int = 50,
    seed: int = 0,
    model_type: str | None = None,
) -> jnp.ndarray:
    """Return probabilities of shape [S, N, C].

    S = n_samples for stochastic models, 1 for deterministic ones.
    """
    model_type = model_type or getattr(model, "model_type", "deterministic")

    if model_type == "deterministic":
        return jnp.asarray(_forward(model, x, sample=False))[None, ...]

    stream = "dropout" if model_type == "mc_dropout" else "bayes"

    out = []
    for s in range(n_samples):
        rngs = nnx.Rngs(**{stream: seed * 100_003 + s})
        out.append(_forward(model, x, rngs=rngs, sample=True))
    return jnp.stack(out, axis=0)


def mc_probs_batched(
    model,
    x: np.ndarray,
    *,
    n_samples: int = 50,
    seed: int = 0,
    batch_size: int = 500,
    model_type: str | None = None,
) -> np.ndarray:
    """Memory-safe version over a full split. Returns np array [S, N, C]."""
    chunks = []
    for i in range(0, len(x), batch_size):
        xb = jnp.asarray(x[i : i + batch_size])
        chunks.append(
            np.asarray(
                mc_probs(model, xb, n_samples=n_samples, seed=seed,
                         model_type=model_type)
            )
        )
    return np.concatenate(chunks, axis=1)


def predictive_mean(probs: np.ndarray) -> np.ndarray:
    """[S, N, C] -> [N, C]."""
    return np.asarray(probs).mean(axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(pbar: np.ndarray, y: np.ndarray) -> float:
    return float((pbar.argmax(axis=-1) == y).mean())


def nll(pbar: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p_true = pbar[np.arange(len(y)), y]
    return float(-np.log(np.clip(p_true, eps, None)).mean())


def brier(pbar: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier: mean over inputs of sum_c (p_c - 1[y=c])^2, range [0,2].

    Some papers halve this or report only the true-class term -- state which
    convention the thesis uses.
    """
    onehot = np.zeros_like(pbar)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((pbar - onehot) ** 2).sum(axis=-1).mean())


def ece(
    pbar: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 15,
    adaptive: bool = False,
) -> dict:
    """ECE on top-1 confidence, plus the per-bin table for reliability plots.

    `adaptive=True` uses equal-mass (quantile) bins. Equal-width ECE is
    unstable when nearly all confidences pile up near 1.0 -- exactly what
    happens on MNIST -- so report both.
    """
    conf = pbar.max(axis=-1)
    correct = (pbar.argmax(axis=-1) == y).astype(np.float64)
    n = len(y)

    if adaptive:
        edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0 + 1e-12
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    total = 0.0
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            table.append({"lo": float(lo), "hi": float(hi), "count": 0,
                          "acc": None, "conf": None})
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(conf[mask].mean())
        total += (count / n) * abs(bin_acc - bin_conf)
        table.append({"lo": float(lo), "hi": float(hi), "count": count,
                      "acc": bin_acc, "conf": bin_conf})

    return {"ece": float(total), "bins": table}


# ---------------------------------------------------------------------------
# Uncertainty decomposition
# ---------------------------------------------------------------------------

def uncertainty_measures(probs: np.ndarray, eps: float = 1e-12) -> dict:
    """probs [S, N, C] -> dict of per-input arrays, each [N]."""
    probs = np.asarray(probs, dtype=np.float64)
    pbar = probs.mean(axis=0)

    total = -(pbar * np.log(np.clip(pbar, eps, None))).sum(axis=-1)
    cond = -(probs * np.log(np.clip(probs, eps, None))).sum(axis=-1).mean(axis=0)
    mi = total - cond

    return {
        "predictive_entropy": total,
        "aleatoric_entropy": cond,
        "mutual_information": np.maximum(mi, 0.0),   # MC noise can dip below 0
        "expected_variance": probs.var(axis=0).sum(axis=-1),
        "confidence": pbar.max(axis=-1),
    }


def clean_report(probs: np.ndarray, y: np.ndarray, *, n_bins: int = 15) -> dict:
    """The Phase B 'calibration baselines on clean data' row."""
    pbar = predictive_mean(probs)
    unc = uncertainty_measures(probs)
    ece_ew = ece(pbar, y, n_bins=n_bins)
    ece_ad = ece(pbar, y, n_bins=n_bins, adaptive=True)
    return {
        "n": int(len(y)),
        "mc_samples": int(probs.shape[0]),
        "accuracy": accuracy(pbar, y),
        "nll": nll(pbar, y),
        "brier": brier(pbar, y),
        "ece": ece_ew["ece"],
        "ece_adaptive": ece_ad["ece"],
        # Per-bin (count, acc, conf) tables: enable reliability diagrams and
        # the binned Brier calibration/refinement decomposition downstream,
        # which the scalar Brier alone cannot support.
        "reliability_bins": ece_ew["bins"],
        "reliability_bins_adaptive": ece_ad["bins"],
        "mean_predictive_entropy": float(unc["predictive_entropy"].mean()),
        "mean_mutual_information": float(unc["mutual_information"].mean()),
        # Aleatoric component. Equals mean_predictive_entropy -
        # mean_mutual_information by definition; emitted directly so
        # future runs do not depend on the reconstruction in
        # analysis.loading. NOTE: uncertainty_measures names this key
        # "aleatoric_entropy", not "expected_entropy".
        "mean_expected_entropy": float(unc["aleatoric_entropy"].mean()),
        "mean_confidence": float(unc["confidence"].mean()),
    }