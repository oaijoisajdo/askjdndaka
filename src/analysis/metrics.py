"""
Statistical primitives, numpy only.

Deliberately dependency-free (no scipy, no sklearn): the analysis stage should
run anywhere the JSONs land, including a machine without the JAX environment.
All functions return NaN rather than raising when a statistic is undefined --
a seed with zero adversarial errors has no AUROC, and that seed must be
dropped from the mean rather than crash the table build. Count the NaNs with
``n_valid`` when summarizing.
"""

from __future__ import annotations

import numpy as np

NAN = float("nan")


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Equivalent to scipy's rankdata."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # Average within tie groups.
    sorted_a = a[order]
    start = 0
    for i in range(1, len(a) + 1):
        if i == len(a) or sorted_a[i] != sorted_a[start]:
            ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def _clean_pair(score, label):
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=float)
    ok = np.isfinite(score) & np.isfinite(label)
    return score[ok], label[ok].astype(int)


def is_constant(score) -> bool:
    """
    True when a score carries no ranking information at all.

    Matters for the deterministic family, where mutual information is
    identically zero. The tie-aware rank statistic would return exactly 0.5
    and average precision exactly the prevalence -- both look like valid
    no-skill measurements and would plot as a flat line that reads as a
    finding rather than a degeneracy.
    """
    s = np.asarray(score, dtype=float)
    s = s[np.isfinite(s)]
    return len(s) == 0 or bool(np.all(s == s[0]))


def auroc(score, label) -> float:
    """
    P(score of a positive > score of a negative), ties counted as half.

    label == 1 marks the event being detected (an adversarial error, or a
    certification failure). Computed from the rank statistic, so ties are
    handled correctly -- which matters because confidence saturates at 1.0
    for many inputs and produces large tie groups. A wholly constant score
    returns NaN, not 0.5; see ``is_constant``.
    """
    s, y = _clean_pair(score, label)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0 or is_constant(s):
        return NAN
    r = _rank(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(score, label) -> float:
    """
    Average precision. Report alongside ``prevalence``: the no-skill baseline
    is prevalence itself, and attack success moves sharply with epsilon, so a
    raw AUPRC is uninterpretable without it. NaN for a constant score.
    """
    s, y = _clean_pair(score, label)
    n_pos = int(y.sum())
    if n_pos == 0 or len(y) == 0 or is_constant(s):
        return NAN

    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]

    ap, tp, seen, prev_recall = 0.0, 0, 0, 0.0
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:      # process a whole tie group
            j += 1
        tp += int(y[i:j].sum())
        seen = j
        recall = tp / n_pos
        ap += (recall - prev_recall) * (tp / seen)
        prev_recall = recall
        i = j
    return float(ap)


def prevalence(label) -> float:
    y = np.asarray(label, dtype=float)
    y = y[np.isfinite(y)]
    return float(y.mean()) if len(y) else NAN


def spearman(x, y) -> float:
    """Pearson correlation of average ranks. NaN if either side is constant."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return NAN
    rx, ry = _rank(x[ok]), _rank(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return NAN
    return float(np.corrcoef(rx, ry)[0, 1])


def ece(confidence, correct, n_bins: int = 15) -> float:
    """
    Expected calibration error, equal-width confidence bins.

    A distribution-level statistic: compute it once over all inputs in a seed.
    Never average per-input values, and never average ECE across seeds before
    computing it within them.
    """
    conf = np.asarray(confidence, dtype=float)
    acc = np.asarray(correct, dtype=float)
    ok = np.isfinite(conf) & np.isfinite(acc)
    conf, acc = conf[ok], acc[ok]
    if len(conf) == 0:
        return NAN

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def reliability_bins(confidence, correct, n_bins: int = 15) -> list[dict]:
    """Per-bin counts behind the ECE, for reliability diagrams."""
    conf = np.asarray(confidence, dtype=float)
    acc = np.asarray(correct, dtype=float)
    ok = np.isfinite(conf) & np.isfinite(acc)
    conf, acc = conf[ok], acc[ok]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin": b,
            "bin_lower": float(edges[b]),
            "bin_upper": float(edges[b + 1]),
            "count": int(m.sum()),
            "mean_confidence": float(conf[m].mean()) if m.any() else NAN,
            "accuracy": float(acc[m].mean()) if m.any() else NAN,
        })
    return rows


def risk_coverage(score, error, coverages) -> list[dict]:
    """
    Selective risk when the ``coverage`` least-risky inputs are retained.

    Abstention is the operational reason to care about uncertainty at all, so
    this is the criterion the plan puts above "did uncertainty go up".
    """
    s, e = _clean_pair(score, error)
    if len(s) == 0:
        return [{"coverage": float(c), "selective_risk": NAN, "n_retained": 0}
                for c in coverages]

    order = np.argsort(s, kind="mergesort")     # ascending risk
    e_sorted = e[order]
    rows = []
    for c in coverages:
        k = max(1, int(np.ceil(float(c) * len(s))))
        rows.append({
            "coverage": float(c),
            "selective_risk": float(e_sorted[:k].mean()),
            "n_retained": int(k),
        })
    return rows


def normalized_auc(x, y) -> float:
    """
    Trapezoidal area under y(x), divided by the x-range.

    Normalizing keeps the number on the scale of the underlying metric (an
    accuracy AUC of 0.6 reads as "0.6 average accuracy over the grid") and
    makes it comparable if the epsilon grid ever changes.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return NAN
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = x[-1] - x[0]
    return float(np.trapezoid(y, x) / span) if span > 0 else NAN


def summarize_across(values) -> dict:
    """
    Mean, SD, seed SE and a 95% t interval over the seed-level replicates.

    NaNs are dropped and counted: ``n_valid`` is the number of seeds where the
    statistic was defined, which the plan requires reporting whenever a
    correlation or AUROC can be undefined.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"mean": NAN, "sd": NAN, "se": NAN,
                "ci_low": NAN, "ci_high": NAN, "n_valid": 0}

    mean, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else NAN
    half = _T95.get(n - 1, 1.96) * se if n > 1 else NAN
    return {
        "mean": mean, "sd": sd, "se": se,
        "ci_low": mean - half if n > 1 else NAN,
        "ci_high": mean + half if n > 1 else NAN,
        "n_valid": n,
    }


# Two-sided 95% t quantiles by degrees of freedom; small-n seed counts make
# the normal approximation noticeably anti-conservative (n=3 -> 4.30, not 1.96).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093, 24: 2.064, 29: 2.045}
