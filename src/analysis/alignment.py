"""
Uncertainty--certification alignment analysis on per_input.csv.

Implements the analysis-side fixes agreed for section E, adapted to the
repository's per_input schema and to the frozen-protocol baseline:

  1. Bracketed Spearman association against pessimistic/optimistic P_safe
     (no exclusion of inconclusive outcomes), Fisher-z averaged across seeds.
  2. Three-valued certification labels with direction-valid bounds:
        robust : Clopper-Pearson lower (pessimistic estimator) >= tau
        below  : theta-margin upper bound (optimistic estimator) <  tau
        inconclusive : rest
     plus the inconclusive-vs-conclusive diagnostic AUROC.
  3. Nested incremental model per (family, condition, seed, eps):
        M0: one_minus_margin_clean + one_minus_confidence_clean
        M1: M0 + mutual_information_clean
     LPO-CV AUROC on robust-vs-below, Delta AUROC, and the PRIMARY
     ENDPOINT partial Spearman rho(MI, P_safe | margin), with the
     margin+confidence conditioning set as a sensitivity.

Schema notes (this repository's per_input.csv):
  - columns are unsuffixed (``mutual_information``, ``confidence``, ...) and
    the run keys are ``training`` and ``eval_epsilon``; ``load_per_input``
    normalizes them to the internal ``*_clean`` / ``condition`` / ``eps``
    names so the analysis code reads identically to the plan.
  - ``predictive_margin`` is the top-2 posterior-predictive margin exported
    by experiments/uncertainty.py; ``one_minus_margin_clean`` is its
    uncertainty-direction transform. Builds that predate the margin export
    carry NaN there; ``incremental`` refuses to silently fall back --
    pass ``m0=("one_minus_confidence_clean",)`` explicitly to reproduce the
    pilot analysis on such builds, and label the output as pilot.

Semantics notes baked in:
  - The CP interval in the CSV belongs to the PESSIMISTIC estimator only
    (verified: its k reconstructs to machine precision under n_samples_used;
    the optimistic side stopped at a different n and is not reconstructable).
    After the A2 export fix, both sides' (n, k, unk) and the optimistic CP
    interval are exported directly and no reconstruction is needed.
  - theta = 0.075 puts a hard ceiling of 1 - theta = 0.925 on p_safe_lower,
    so tau = 0.95 is unreachable by construction under the theta bounds.
  - The deterministic family has MI == 0 up to float noise and is excluded
    from incremental fits; it serves as the margin-only reference family.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RUN = ["family", "condition", "seed"]
CELL = RUN + ["eps"]

CLEAN_SCORES = [
    "mutual_information_clean",
    "predictive_entropy_clean",
    "expected_entropy_clean",
    "one_minus_confidence_clean",
    "one_minus_margin_clean",
]
BASELINE = "one_minus_margin_clean"            # frozen-protocol baseline
M0_COVARS = (BASELINE, "one_minus_confidence_clean")
MI = "mutual_information_clean"

_RENAME = {
    "training": "condition",
    "eval_epsilon": "eps",
    "clean_correct": "correct_clean",
    "mutual_information": "mutual_information_clean",
    "predictive_entropy": "predictive_entropy_clean",
    "expected_entropy": "expected_entropy_clean",
}


def load_per_input(path_or_df) -> pd.DataFrame:
    """Normalize the repository per_input schema to the analysis schema."""
    d = (path_or_df.copy() if isinstance(path_or_df, pd.DataFrame)
         else pd.read_csv(path_or_df))
    d = d.rename(columns={k: v for k, v in _RENAME.items()
                          if k in d.columns and v not in d.columns})
    if "confidence" in d.columns and "one_minus_confidence_clean" not in d.columns:
        d["one_minus_confidence_clean"] = 1.0 - d["confidence"]
    if "one_minus_confidence_clean" not in d.columns:
        d["one_minus_confidence_clean"] = np.nan
    if "one_minus_margin_clean" not in d.columns:
        if "predictive_margin" in d.columns:
            d["one_minus_margin_clean"] = 1.0 - d["predictive_margin"]
        else:
            d["one_minus_margin_clean"] = np.nan
    d["correct_clean"] = d["correct_clean"].astype(float)
    return d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fisher_mean(rhos: pd.Series) -> tuple[float, float, int]:
    """Fisher-z average of correlations; returns (mean_rho, se_rho, n)."""
    r = rhos.dropna().clip(-0.999999, 0.999999)
    n = len(r)
    if n == 0:
        return np.nan, np.nan, 0
    z = np.arctanh(r)
    zm, zse = z.mean(), (z.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)
    rm = np.tanh(zm)
    # delta-method SE back on the rho scale
    rse = (1 - rm**2) * zse if n > 1 else np.nan
    return float(rm), float(rse), n


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    if m.sum() < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return np.nan
    return float(spearmanr(x[m], y[m]).statistic)


def _safe_auroc(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label)
    m = np.isfinite(score)
    if m.sum() < 3 or len(np.unique(label[m])) < 2:
        return np.nan
    return float(roc_auc_score(label[m], score[m]))


# ---------------------------------------------------------------------------
# 1. bracketed association (no exclusion)
# ---------------------------------------------------------------------------

def bracketed_association(cert: pd.DataFrame,
                          scores=CLEAN_SCORES) -> pd.DataFrame:
    """Seed-level Spearman vs both P_safe estimators, then Fisher-z summary.

    The (pessimistic, optimistic) pair brackets the association under any
    resolution of unknown verifier outcomes: nothing is excluded.
    """
    rows = []
    for keys, g in cert.groupby(CELL, dropna=False):
        rec = dict(zip(CELL, keys))
        for s in scores:
            rec_s = dict(rec, score=s)
            rec_s["neg_rho_pess"] = -_safe_spearman(g[s], g["p_safe_point"])
            rec_s["neg_rho_opt"] = -_safe_spearman(
                g[s], g["p_safe_point_optimistic"])
            rows.append(rec_s)
    seed_level = pd.DataFrame(rows)

    out = []
    grp = ["family", "condition", "eps", "score"]
    for keys, g in seed_level.groupby(grp):
        rec = dict(zip(grp, keys))
        for side in ("pess", "opt"):
            m, se, n = fisher_mean(g[f"neg_rho_{side}"])
            rec[f"neg_rho_{side}"] = m
            rec[f"neg_rho_{side}_se"] = se
        rec["n_seeds"] = n
        out.append(rec)
    return pd.DataFrame(out), seed_level


# ---------------------------------------------------------------------------
# 2. direction-valid three-way labels + inconclusive diagnostic
# ---------------------------------------------------------------------------

def add_labels(cert: pd.DataFrame, tau: float) -> pd.DataFrame:
    """robust / below / inconclusive with each direction on its valid bound."""
    d = cert.copy()
    d["lab_robust"] = d["p_safe_ci_low"] >= tau          # CP, pessimistic
    d["lab_below"] = d["p_safe_upper"] < tau             # theta, optimistic
    overlap = d["lab_robust"] & d["lab_below"]
    if overlap.any():                                    # cannot happen if
        d.loc[overlap, ["lab_robust", "lab_below"]] = False   # bounds sane
    d["lab_inconclusive"] = ~(d["lab_robust"] | d["lab_below"])
    d["cert_label"] = np.select(
        [d["lab_robust"], d["lab_below"]], ["robust", "below"], "inconclusive")
    return d


def label_analyses(cert: pd.DataFrame, tau: float,
                   scores=CLEAN_SCORES) -> pd.DataFrame:
    """Per-cell: class shares, ordinal association, diagnostic AUROC, and
    conclusive-only AUROC bracketed by conservative inconclusive assignment."""
    d = add_labels(cert, tau)
    ordinal = d["cert_label"].map({"below": 0, "inconclusive": 1, "robust": 2})
    d["_ord"] = ordinal
    rows = []
    for keys, g in d.groupby(CELL, dropna=False):
        rec = dict(zip(CELL, keys), tau=tau,
                   frac_robust=g["lab_robust"].mean(),
                   frac_below=g["lab_below"].mean(),
                   frac_inconclusive=g["lab_inconclusive"].mean())
        conclusive = ~g["lab_inconclusive"]
        for s in scores:
            r = dict(rec, score=s)
            # ordinal: does risk fall as label rises below->inconcl->robust
            r["neg_rho_ordinal"] = -_safe_spearman(g[s], g["_ord"])
            # diagnostic: does the score predict inconclusiveness itself
            r["auroc_inconclusive"] = _safe_auroc(
                g[s].to_numpy(), g["lab_inconclusive"].astype(int).to_numpy())
            # conclusive-only AUROC (below = positive) + conservative brackets
            r["auroc_conclusive"] = _safe_auroc(
                g.loc[conclusive, s].to_numpy(),
                g.loc[conclusive, "lab_below"].astype(int).to_numpy())
            r["auroc_inc_as_below"] = _safe_auroc(
                g[s].to_numpy(),
                (g["lab_below"] | g["lab_inconclusive"]).astype(int).to_numpy())
            r["auroc_inc_as_robust"] = _safe_auroc(
                g[s].to_numpy(), g["lab_below"].astype(int).to_numpy())
            r["n_conclusive"] = int(conclusive.sum())
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. nested incremental model
# ---------------------------------------------------------------------------

def _lpo_auroc(X: np.ndarray, y: np.ndarray, max_pairs: int = 400,
               rng_seed: int = 0) -> float:
    """Leave-pair-out cross-validated AUROC (Airola et al. 2011).

    Unbiased for small n and immune to the LOO intercept-inversion
    pathology under class imbalance: each (positive, negative) pair is
    held out jointly and scored by whether the model ranks them correctly.
    Ties count 0.5. Pairs are subsampled beyond `max_pairs`.
    """
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    pairs = [(i, j) for i in pos for j in neg]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(rng_seed)
        pairs = [pairs[k] for k in
                 rng.choice(len(pairs), max_pairs, replace=False)]
    idx = np.arange(len(y))
    wins = 0.0
    for i, j in pairs:
        tr = (idx != i) & (idx != j)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        pi, pj = clf.predict_proba(X[[i, j]])[:, 1]
        wins += 1.0 if pi > pj else (0.5 if pi == pj else 0.0)
    return wins / len(pairs)


def partial_spearman(x: pd.Series, y: pd.Series,
                     z: pd.Series | pd.DataFrame) -> float:
    """rho(x, y | z) on ranks via residualization.

    ``z`` may be a single Series or a DataFrame of several conditioning
    covariates (each is rank-transformed before residualization).
    """
    z = z.to_frame() if isinstance(z, pd.Series) else z
    m = x.notna() & y.notna() & z.notna().all(axis=1)
    if m.sum() < 5:
        return np.nan
    rx, ry = rankdata(x[m]), rankdata(y[m])
    rz = np.column_stack([rankdata(z.loc[m, c]) for c in z.columns])

    def resid(a, b):
        b = np.column_stack([np.ones(len(a)), b])
        coef, *_ = np.linalg.lstsq(b, a, rcond=None)
        return a - b @ coef

    ex, ey = resid(rx, rz), resid(ry, rz)
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def incremental(cert: pd.DataFrame, tau: float,
                clean_correct_only: bool = True,
                m0: Sequence[str] = M0_COVARS) -> pd.DataFrame:
    """M0 (margin + confidence) vs M1 (+MI) per cell, robust-vs-below labels.

    Deterministic family excluded: MI is zero to float noise there, so the
    fit is vacuous; that family is the margin-only reference by construction.

    PRIMARY ENDPOINT: ``partial_rho_mi`` = rho(MI, P_safe | margin), i.e.
    conditioning on BASELINE alone. ``partial_rho_mi_full`` conditions on
    the whole M0 set as a sensitivity.

    Raises if any M0 covariate is entirely NaN (pre-margin builds): pass
    ``m0=("one_minus_confidence_clean",)`` explicitly to run the pilot
    configuration, and label the output as pilot.
    """
    m0 = list(m0)
    d = add_labels(cert, tau)
    d = d[d["family"] != "deterministic"]
    if clean_correct_only:
        d = d[d["correct_clean"] == 1]
    dead = [c for c in m0 if d[c].isna().all()]
    if dead:
        raise ValueError(
            f"M0 covariates entirely NaN in this build: {dead}. "
            "Re-run the export, or pass m0=('one_minus_confidence_clean',) "
            "to reproduce the pilot configuration explicitly.")
    rows = []
    for keys, g in d.groupby(CELL, dropna=False):
        conclusive = ~g["lab_inconclusive"]
        gc = g[conclusive].dropna(subset=m0 + [MI])
        y = gc["lab_below"].astype(int).to_numpy()
        rec = dict(zip(CELL, keys), tau=tau,
                   n=len(gc), prevalence=y.mean() if len(y) else np.nan)
        X0 = gc[m0].to_numpy()
        X1 = gc[m0 + [MI]].to_numpy()
        rec["auroc_m0"] = _lpo_auroc(X0, y)
        rec["auroc_m1"] = _lpo_auroc(X1, y)
        rec["delta_auroc"] = rec["auroc_m1"] - rec["auroc_m0"]
        rec["auroc_mi_alone"] = _safe_auroc(gc[MI].to_numpy(), y)
        rec["partial_rho_mi"] = partial_spearman(          # primary endpoint
            g[MI], -g["p_safe_point"], g[m0[0]])
        rec["partial_rho_mi_full"] = partial_spearman(     # sensitivity
            g[MI], -g["p_safe_point"], g[m0])
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_incremental(inc: pd.DataFrame) -> pd.DataFrame:
    grp = ["family", "condition", "eps", "tau"]
    out = []
    for keys, g in inc.groupby(grp):
        rec = dict(zip(grp, keys), n_seeds=len(g))
        for c in ("auroc_m0", "auroc_m1", "delta_auroc", "auroc_mi_alone"):
            v = g[c].dropna()
            rec[c] = v.mean() if len(v) else np.nan
            rec[f"{c}_se"] = (v.std(ddof=1) / np.sqrt(len(v))
                              if len(v) > 1 else np.nan)
        for c in ("partial_rho_mi", "partial_rho_mi_full"):
            m, se, _ = fisher_mean(g[c])
            rec[c], rec[f"{c}_se"] = m, se
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 4. empirical power: bootstrap CI width and n extrapolation
# ---------------------------------------------------------------------------

def bootstrap_delta_auroc(cert: pd.DataFrame, tau: float, family: str,
                          condition: str, eps: float, n_boot: int = 1000,
                          seed: int = 0,
                          m0: Sequence[str] = M0_COVARS) -> pd.DataFrame:
    """Input-level bootstrap of Delta AUROC (in-sample fits: the bootstrap
    is used for spread, not level; LPO-CV handles level in `incremental`)."""
    rng = np.random.default_rng(seed)
    m0 = list(m0)
    d = add_labels(cert, tau)
    d = d[(d.family == family) & (d.condition == condition) & (d.eps == eps)
          & (d.correct_clean == 1)]
    rows = []
    for s, g in d.groupby("seed"):
        gc = g[~g["lab_inconclusive"]].dropna(subset=m0 + [MI])
        y = gc["lab_below"].astype(int).to_numpy()
        X0 = gc[m0].to_numpy()
        X1 = gc[m0 + [MI]].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        deltas = []
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yb = y[idx]
            if len(np.unique(yb)) < 2:
                continue
            a0 = _fit_auroc(X0[idx], yb)
            a1 = _fit_auroc(X1[idx], yb)
            if not (np.isnan(a0) or np.isnan(a1)):
                deltas.append(a1 - a0)
        if deltas:
            deltas = np.array(deltas)
            rows.append({"seed": s, "n_inputs": n,
                         "delta_mean": deltas.mean(),
                         "delta_se_boot": deltas.std(ddof=1),
                         "ci_width_95": np.percentile(deltas, 97.5)
                                        - np.percentile(deltas, 2.5)})
    return pd.DataFrame(rows)


def _fit_auroc(X: np.ndarray, y: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    return _safe_auroc(clf.predict_proba(X)[:, 1], y)


def required_n(se_now: float, n_now: int, target_delta: float = 0.05,
               power_z: float = 2.8) -> float:
    """n for 80% power on target_delta at alpha=.05, SE ~ 1/sqrt(n)."""
    se_target = target_delta / power_z
    return n_now * (se_now / se_target) ** 2
