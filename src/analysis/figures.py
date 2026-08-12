"""
Thesis figures: descriptive layer (D1-D4) then alignment layer (A1-A7).

Descriptive figures read the model-level tables (clean_metrics.csv,
pgd_metrics.csv: full eval split, n=2000) and answer "what do these models
do under attack" before any alignment claim. Alignment figures read the
normalized per-input table (certification subset, n=50).

All functions return a Figure and regenerate unchanged on confirmatory data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analysis import alignment as al
from analysis import seed_stats as ss
from analysis import metrics as M

FAMS = ["vogn", "bbb", "mc_dropout", "deterministic"]
FLAB = {"vogn": "VOGN", "bbb": "BBB", "mc_dropout": "MC-dropout",
        "deterministic": "Deterministic"}
FCOL = {"vogn": "#0072B2", "bbb": "#E69F00", "mc_dropout": "#009E73",
        "deterministic": "#666666"}
LCOL = {"robust": "#0072B2", "inconclusive": "#C7C7C7", "below": "#D55E00"}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.8, "lines.markersize": 5,
    "legend.frameon": False,
})


def load_model_metrics(tables_dir: str):
    """(clean, pgd) model-level frames from the current build_tables output.

    ``build_tables`` writes a single ``eval_metrics.csv`` in which eps == 0 is
    the clean block and eps > 0 are the attacked blocks. Older snapshots split
    these into clean_metrics.csv / pgd_metrics.csv; both layouts are accepted.

    ``mean_expected_entropy`` is derived here too if absent, so figures work
    on tables built before the loader fix.
    """
    root = Path(tables_dir)
    if (root / "eval_metrics.csv").exists():
        em = pd.read_csv(root / "eval_metrics.csv")
        if "training" in em.columns and "condition" not in em.columns:
            em = em.rename(columns={"training": "condition"})
        if "eval_epsilon" in em.columns and "eps" not in em.columns:
            em = em.rename(columns={"eval_epsilon": "eps"})
        if ("mean_expected_entropy" not in em.columns
                and {"mean_predictive_entropy",
                     "mean_mutual_information"} <= set(em.columns)):
            em["mean_expected_entropy"] = (em["mean_predictive_entropy"]
                                           - em["mean_mutual_information"])
        cm = em[em["eps"] == 0.0].copy()
        pm = em[em["eps"] > 0.0].copy()
        return cm, pm

    cm = pd.read_csv(root / "clean_metrics.csv").rename(
        columns={"training": "condition"})
    pm = pd.read_csv(root / "pgd_metrics.csv").rename(
        columns={"training": "condition", "eval_epsilon": "eps"})
    return cm, pm


def cert_rows(cert: pd.DataFrame) -> pd.DataFrame:
    """Rows where the verifier actually ran.

    per_input.csv spans 7 radii but certification ran on 5; the PGD-only
    radii (0.03, 0.15) are all-NaN in every certification column. Passing
    them into a certification figure turns ``NaN >= tau`` into False, which
    labels the whole radius "inconclusive" and plots a spurious collapse
    (lo = 0, hi = 1). Every certification figure must go through here.
    """
    return cert[cert["p_safe_point"].notna()].copy()


def _seed_mean(g, col):
    v = g.groupby("seed")[col].mean()
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1
                      else np.nan)


def _fam_legend(ax, fams=FAMS, **kw):
    ax.legend([Line2D([], [], color=FCOL[f]) for f in fams],
              [FLAB[f] for f in fams], **kw)


def _curve_panels(cm, pm, col, ylabel, title, clean_at_zero=True,
                  fams=FAMS, ylim=None):
    """Two panels (robust/standard): metric vs eps, clean value at eps=0."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, cond in zip(axes, ["robust", "standard"]):
        for fam in fams:
            xs, ys, es = [], [], []
            if clean_at_zero:
                m, se = _seed_mean(cm[(cm.family == fam)
                                      & (cm.condition == cond)], col)
                xs.append(0.0); ys.append(m); es.append(se)
            for eps in sorted(pm["eps"].unique()):
                g = pm[(pm.family == fam) & (pm.condition == cond)
                       & (pm.eps == eps)]
                if not len(g):
                    continue
                m, se = _seed_mean(g, col)
                xs.append(eps); ys.append(m); es.append(se)
            ax.errorbar(xs, ys, yerr=es, color=FCOL[fam], marker="o",
                        capsize=2.5)
        ax.axvline(0.08, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.set_xlabel(r"attack radius $\varepsilon$"
                      + ("  (0 = clean)" if clean_at_zero else ""))
        ax.set_title(f"{cond}-trained")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    _fam_legend(axes[0], fams=fams, loc="best")
    fig.suptitle(title, fontsize=12)
    return fig


# ------------------------------------------------------------- descriptive
def fig_d1_accuracy(cm, pm):
    """D1: clean and PGD accuracy over radius. The basic robustness result:
    what robust training buys, what standard training loses, and where the
    families separate before any uncertainty question is asked."""
    return _curve_panels(
        cm, pm, "accuracy", "accuracy (eval split, n=2000)",
        "Predictive accuracy under EOT-PGD "
        r"(dotted: $\varepsilon_{\mathrm{train}}$)", ylim=(0, 1.0))


def fig_d2_uncertainty(cm, pm):
    """D2: mean MI and predictive entropy over radius. Does epistemic
    uncertainty respond to attack at all, per family? The precondition for
    any alignment claim: a flat MI curve here would end the thesis early."""
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True)
    for row, (col, lab) in enumerate([
            ("mean_mutual_information", "mean mutual information"),
            ("mean_predictive_entropy", "mean predictive entropy")]):
        for ax, cond in zip(axes[row], ["robust", "standard"]):
            for fam in [f for f in FAMS if f != "deterministic"]:
                xs, ys, es = [0.0], [], []
                m, se = _seed_mean(cm[(cm.family == fam)
                                      & (cm.condition == cond)], col)
                ys.append(m); es.append(se)
                for eps in sorted(pm["eps"].unique()):
                    g = pm[(pm.family == fam) & (pm.condition == cond)
                           & (pm.eps == eps)]
                    m, se = _seed_mean(g, col)
                    xs.append(eps); ys.append(m); es.append(se)
                ax.errorbar(xs, ys, yerr=es, color=FCOL[fam], marker="o",
                            capsize=2.5)
            ax.axvline(0.08, color="k", lw=0.7, ls=":", alpha=0.6)
            if row == 0:
                ax.set_title(f"{cond}-trained")
            if row == 1:
                ax.set_xlabel(r"attack radius $\varepsilon$  (0 = clean)")
        axes[row][0].set_ylabel(lab)
    _fam_legend(axes[0][0],
                fams=[f for f in FAMS if f != "deterministic"], loc="best")
    fig.suptitle("Uncertainty response to attack (eval split)", fontsize=12)
    return fig


def fig_d3_calibration(cm, pm):
    """D3: calibration under attack: ECE and Brier over radius. Clean
    calibration is near-perfect everywhere; the question is how fast it
    degrades and whether robust training changes the slope."""
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True)
    for row, (col, lab) in enumerate([("ece", "ECE (15 bins)"),
                                      ("brier", "Brier score")]):
        for ax, cond in zip(axes[row], ["robust", "standard"]):
            for fam in FAMS:
                xs, ys, es = [0.0], [], []
                m, se = _seed_mean(cm[(cm.family == fam)
                                      & (cm.condition == cond)], col)
                ys.append(m); es.append(se)
                for eps in sorted(pm["eps"].unique()):
                    g = pm[(pm.family == fam) & (pm.condition == cond)
                           & (pm.eps == eps)]
                    m, se = _seed_mean(g, col)
                    xs.append(eps); ys.append(m); es.append(se)
                ax.errorbar(xs, ys, yerr=es, color=FCOL[fam], marker="o",
                            capsize=2.5)
            ax.axvline(0.08, color="k", lw=0.7, ls=":", alpha=0.6)
            if row == 0:
                ax.set_title(f"{cond}-trained")
            if row == 1:
                ax.set_xlabel(r"attack radius $\varepsilon$  (0 = clean)")
        axes[row][0].set_ylabel(lab)
    _fam_legend(axes[0][0], loc="upper left")
    fig.suptitle("Calibration under attack (eval split)", fontsize=12)
    return fig


def fig_d4_certified(cert, tau=0.9):
    """D4: certified-robust share over radius, bracketed. The bridge from
    descriptive to alignment: the verified counterpart of D1, with the
    verifier's unknowns carried as an interval instead of resolved."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    d = al.add_labels(cert_rows(cert), tau)
    for ax, cond in zip(axes, ["robust", "standard"]):
        for fam in FAMS:
            xs, lo, hi = [], [], []
            for eps in sorted(d["eps"].unique()):
                g = d[(d.family == fam) & (d.condition == cond)
                      & (d.eps == eps)]
                if not len(g):
                    continue
                xs.append(eps)
                lo.append(g["lab_robust"].mean())                  # unknowns fail
                hi.append((g["lab_robust"]
                           | g["lab_inconclusive"]).mean())        # unknowns pass
            ax.fill_between(xs, lo, hi, color=FCOL[fam], alpha=0.18, lw=0)
            ax.plot(xs, lo, color=FCOL[fam], marker="o")
            ax.plot(xs, hi, color=FCOL[fam], ls="--", lw=1.1, alpha=0.7)
        ax.set_xlabel(r"certification radius $\varepsilon$")
        ax.set_title(f"{cond}-trained")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel(f"certified-robust share (τ = {tau:g})")
    handles = ([Line2D([], [], color=FCOL[f]) for f in FAMS]
               + [Line2D([], [], color="k", marker="o"),
                  Line2D([], [], color="k", ls="--", lw=1.1, alpha=0.7)])
    axes[1].legend(handles, [FLAB[f] for f in FAMS]
                   + ["unknowns → fail", "unknowns → pass"],
                   loc="upper right", fontsize=8.5)
    fig.suptitle("Certified robustness (50-input subset, bracketed)",
                 fontsize=12)
    return fig


def fig_d4b_bound_comparison(cert, tau=0.9):
    """D4b: certified share under the fixed-n Clopper-Pearson lower limit
    vs the sequentially-valid Massart theta-bound (p_safe_lower).

    The sequential scheme's terminal guarantee is p_hat +/- theta at
    confidence 1 - gamma per side; the CP interval only steers the adaptive
    stopping rule and is computed at a data-dependent n. Where the two
    curves diverge, the divergence is the bound choice, not the models.
    """
    d = cert_rows(cert)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, cond in zip(axes, ["robust", "standard"]):
        for fam in FAMS:
            g0 = d[(d.family == fam) & (d.condition == cond)]
            for col, ls, mk in (("p_safe_ci_low", "-", "o"),
                                ("p_safe_lower", "--", "s")):
                s = (g0.assign(ok=g0[col] >= tau)
                       .groupby("eps")["ok"].mean())
                ax.plot(s.index, s.values, color=FCOL[fam], ls=ls,
                        marker=mk, ms=4, alpha=1.0 if ls == "-" else 0.65)
        ax.set_xlabel(r"certification radius $\varepsilon$")
        ax.set_title(f"{cond}-trained")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel(f"share with lower bound ≥ τ = {tau:g}")
    handles = ([Line2D([], [], color=FCOL[f]) for f in FAMS]
               + [Line2D([], [], color="k", marker="o", ms=4),
                  Line2D([], [], color="k", ls="--", marker="s", ms=4,
                         alpha=0.65)])
    axes[1].legend(handles, [FLAB[f] for f in FAMS]
                   + ["Clopper–Pearson (fixed-n)",
                      "Massart θ-bound (sequential)"],
                   loc="upper right", fontsize=8)
    fig.suptitle("Robust-gate sensitivity to the bound choice", fontsize=12)
    return fig


# --------------------------------------------------------------- alignment
def fig_a1b_forest(cert, baseline, eps=0.08):
    """A1b: the primary endpoint as a forest plot at the pre-registered
    radius. Small dots = per-seed partial rho; large markers = Fisher-z
    mean +/- SE; filled = pessimistic, open = optimistic estimator.

    H1 reads off directly: VOGN's [pessimistic, optimistic] bracket must
    exclude zero and the top-down ordering VOGN > BBB > MC-dropout hold.
    The over-radius companion (A1) moves to a supporting role.
    """
    cert = cert_rows(cert)
    fams = ["vogn", "bbb", "mc_dropout"]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    rng = np.random.default_rng(0)
    for i, fam in enumerate(fams):
        y0 = len(fams) - 1 - i
        for k, (side, col) in enumerate(
                (("pessimistic", "p_safe_point"),
                 ("optimistic", "p_safe_point_optimistic"))):
            dy = 0.17 - 0.34 * k
            rhos = []
            for seed in sorted(cert["seed"].unique()):
                g = cert[(cert.family == fam)
                         & (cert.condition == "robust")
                         & (cert.eps == eps) & (cert.seed == seed)
                         & (cert.correct_clean == 1)]
                if len(g):
                    rhos.append(al.partial_spearman(
                        g[al.MI], -g[col], g[baseline]))
            m, se, n = al.fisher_mean(pd.Series(rhos, dtype=float))
            jit = rng.uniform(-0.04, 0.04, len(rhos))
            ax.scatter(rhos, y0 + dy + jit, s=13, color=FCOL[fam],
                       alpha=0.45, lw=0)
            ax.errorbar(m, y0 + dy, xerr=se, color=FCOL[fam],
                        marker="o" if k == 0 else "o",
                        mfc=FCOL[fam] if k == 0 else "white",
                        mec=FCOL[fam], ms=8, capsize=3, lw=1.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([FLAB[f] for f in reversed(fams)])
    ax.set_xlabel(rf"partial $\rho(\mathrm{{MI}}, P_\mathrm{{safe}} \mid "
                  rf"\mathrm{{margin}})$ at $\varepsilon$ = {eps:g}")
    ax.legend([Line2D([], [], marker="o", color="k", ls="", ms=8),
               Line2D([], [], marker="o", color="k", mfc="white", ls="",
                      ms=8),
               Line2D([], [], marker="o", color="0.5", ls="", ms=4,
                      alpha=0.5)],
              ["pessimistic (Fisher-z ± SE)", "optimistic", "per-seed"],
              loc="lower right", fontsize=8)
    ax.set_title("Primary endpoint — robust condition, clean-correct",
                 fontsize=11)
    return fig


def fig_a1_partial_rho(cert, baseline, pilot=False):
    """A1 / PRIMARY. Partial rho(MI, P_safe | baseline), robust condition,
    Fisher-z over seeds, both estimators.

    Certification grid only. Cells where the endpoint is not estimable
    (P_safe constant within every seed, or too few clean-correct rows)
    stay on the axis as an annotated gap; they are results, not blanks.
    """
    cert = cert_rows(cert)
    fams = [f for f in FAMS if f != "deterministic"]
    epss = sorted(cert["eps"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, fam in zip(axes, fams):
        na_eps: set[float] = set()
        for side, col, ls, alpha in (("pessimistic", "p_safe_point", "-", 1.0),
                                     ("optimistic", "p_safe_point_optimistic",
                                      "--", 0.6)):
            xs, ys, es = [], [], []
            for eps in epss:
                rhos = []
                for seed in sorted(cert["seed"].unique()):
                    g = cert[(cert.family == fam)
                             & (cert.condition == "robust")
                             & (cert.eps == eps) & (cert.seed == seed)
                             & (cert.correct_clean == 1)]
                    if len(g):
                        rhos.append(al.partial_spearman(
                            g[al.MI], -g[col], g[baseline]))
                m, se, n_valid = al.fisher_mean(pd.Series(rhos, dtype=float))
                xs.append(eps)                    # keep the full eps axis;
                ys.append(m if n_valid else np.nan)   # NaN = visible gap,
                es.append(se if n_valid else np.nan)  # never bridged
                if not n_valid:
                    na_eps.add(eps)
            ax.errorbar(xs, ys, yerr=es, color=FCOL[fam], marker="o",
                        ls=ls, alpha=alpha, capsize=2.5, label=side)
        for eps in sorted(na_eps):
            ax.plot(eps, 0.0, marker="x", color="0.35", ms=7, mew=1.6,
                    zorder=5, clip_on=False)
            ax.annotate("not estimable\n($P_\\mathrm{safe}$ constant)",
                        (eps, 0.0), textcoords="offset points",
                        xytext=(2, 10), ha="left", fontsize=7.5,
                        color="0.35")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(FLAB[fam])
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_xticks(epss)
        ax.set_xticklabels([f"{e:g}" for e in epss], fontsize=8.5)
    base_lab = "margin" if "margin" in baseline else "confidence"
    axes[0].set_ylabel(
        rf"partial $\rho(\mathrm{{MI}}, P_{{\mathrm{{safe}}}} \mid "
        rf"\mathrm{{{base_lab}}})$")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Alignment beyond the baseline — robust condition, "
                 "clean-correct" + ("   [PILOT: confidence baseline]"
                                    if pilot else ""), fontsize=12)
    return fig


def fig_a2_bracket_auroc(cert, score=al.MI, tau=0.9):
    """A2: conclusive-only AUROC vs the no-exclusion bracket. Diamonds above
    their band = row-dropping inflated apparent detection."""
    cert = cert_rows(cert)
    fams = [f for f in FAMS if f != "deterministic"]
    epss = sorted(cert["eps"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, fam in zip(axes, fams):
        for i, eps in enumerate(epss):
            co, lo, hi = [], [], []
            for seed in sorted(cert["seed"].unique()):
                g = cert[(cert.family == fam) & (cert.condition == "robust")
                         & (cert.eps == eps) & (cert.seed == seed)]
                if not len(g):
                    continue
                s = g[score]
                c, f = ss._cert_labels(g, tau)
                fb, fr, _ = ss._cert_label_bracket(g, tau)
                co.append(M.auroc(s[c], f[c]))
                pair = [M.auroc(s, fb), M.auroc(s, fr)]
                lo.append(np.nanmin(pair)); hi.append(np.nanmax(pair))
            ax.plot([i, i], [np.nanmean(lo), np.nanmean(hi)],
                    color=FCOL[fam], lw=6, alpha=0.35,
                    solid_capstyle="butt")
            ax.plot(i, np.nanmean(co), marker="D", color=FCOL[fam],
                    ms=6, mec="k", mew=0.5)
        ax.axhline(0.5, color="k", lw=0.7, ls=":")
        ax.set_xticks(range(len(epss)))
        ax.set_xticklabels([f"{e:g}" for e in epss])
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_title(FLAB[fam])
    axes[0].set_ylabel(f"AUROC(MI → certified-below, τ={tau:g})")
    axes[2].legend([Line2D([], [], marker="D", color="k", ls="", ms=6),
                    Line2D([], [], color="k", lw=6, alpha=0.35)],
                   ["conclusive-only", "bracket (no rows dropped)"],
                   loc="lower left", fontsize=8.5)
    fig.suptitle("Exclusion bias: what dropping inconclusive rows did",
                 fontsize=12)
    return fig


def fig_a3_conservatism(cert, tau=0.9):
    """A3: two views of verifier conservatism: unknown fraction over radius,
    and AUROC(MI → inconclusive) with its sign flip."""
    cert = cert_rows(cert)
    epss = sorted(cert["eps"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    for fam in FAMS:
        g = (cert[(cert.family == fam) & (cert.condition == "robust")]
             .groupby("eps")["unknown_frac"])
        ax.errorbar(g.mean().index, g.mean().values, yerr=g.sem().values,
                    color=FCOL[fam], marker="o", capsize=2.5)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("mean unknown fraction")
    ax.set_title("Verdict-stream unknowns")
    _fam_legend(ax, loc="upper right")
    ax = axes[1]
    na_cells = []                       # (fam, eps) with AUROC undefined
    for fam in [f for f in FAMS if f != "deterministic"]:
        xs, ys, es = [], [], []
        for eps in epss:
            vals = []
            for seed in sorted(cert["seed"].unique()):
                g = cert[(cert.family == fam) & (cert.condition == "robust")
                         & (cert.eps == eps) & (cert.seed == seed)]
                if len(g):
                    _, _, inc = ss._cert_label_bracket(g, tau)
                    vals.append(M.auroc(g[al.MI], inc.astype(float)))
            vals = [v for v in vals if not np.isnan(v)]
            xs.append(eps)              # eps stays on the axis either way
            if vals:
                ys.append(np.mean(vals))
                es.append(np.std(vals) / np.sqrt(len(vals)))
            else:
                ys.append(np.nan); es.append(np.nan)
                na_cells.append((fam, eps))
        ax.errorbar(xs, ys, yerr=es, color=FCOL[fam], marker="o",
                    capsize=2.5)
    for fam, eps in na_cells:           # single-class cell: no inconclusives
        ax.plot(eps, 0.5, marker="x", color=FCOL[fam], ms=7, mew=1.6,
                zorder=5)
    if na_cells:
        ax.annotate("× = not estimable (no inconclusive rows)",
                    (0.02, 0.03), xycoords="axes fraction",
                    fontsize=7.5, color="0.35")
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.set_xticks(epss)
    ax.set_xticklabels([f"{e:g}" for e in epss], fontsize=8.5)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("AUROC(MI → inconclusive)")
    ax.set_title(f"Unknowns are MI-predictable (τ={tau:g})")
    fig.suptitle("Certificate conservatism, robust condition", fontsize=12)
    return fig


def fig_a4_censoring(cert):
    """A4: floor/ceiling censoring of the pessimistic point estimate."""
    cert = cert_rows(cert)
    epss = sorted(cert["eps"].unique())
    fig, axes = plt.subplots(1, len(epss), figsize=(2.1 * len(epss), 2.9),
                             sharey=True)
    d = cert[cert.condition == "robust"]
    for ax, eps in zip(axes, epss):
        v = d[d.eps == eps]["p_safe_point"].dropna()
        ax.bar([0], [(v == 0).mean()], width=0.14, color=LCOL["below"])
        ax.bar([1], [(v == 1).mean()], width=0.14, color=LCOL["robust"])
        inner = v[(v > 0) & (v < 1)]
        if len(inner):
            h, e = np.histogram(inner, bins=8, range=(0, 1))
            ax.bar((e[:-1] + e[1:]) / 2, h / len(v), width=np.diff(e) * 0.85,
                   color=LCOL["inconclusive"])
        ax.set_title(rf"$\varepsilon$={eps:g}", fontsize=10)
        pin = (v == 0).mean() + (v == 1).mean()
        ax.text(0.5, 0.9, f"{100*pin:.0f}% at 0/1", transform=ax.transAxes,
                ha="center", fontsize=8.5)
        ax.set_xticks([0, 1])
    axes[0].set_ylabel("share of inputs")
    fig.suptitle(r"Censoring of $\hat P_{\mathrm{safe}}$ (pessimistic, "
                 "robust condition)", fontsize=12)
    return fig


def fig_a5_composition(cert, tau=0.9):
    """A5: label composition as small multiples (family x condition)."""
    d = al.add_labels(cert_rows(cert), tau)
    fams = FAMS
    fig, axes = plt.subplots(2, len(fams), figsize=(10.5, 5.2),
                             sharex=True, sharey=True)
    epss = sorted(d["eps"].unique())
    for r, cond in enumerate(["robust", "standard"]):
        for c, fam in enumerate(fams):
            ax = axes[r][c]
            for i, eps in enumerate(epss):
                g = d[(d.family == fam) & (d.condition == cond)
                      & (d.eps == eps)]
                if not len(g):
                    continue
                fr, fi = g["lab_robust"].mean(), g["lab_inconclusive"].mean()
                fb = g["lab_below"].mean()
                ax.bar(i, fr, color=LCOL["robust"], width=0.8)
                ax.bar(i, fi, bottom=fr, color=LCOL["inconclusive"],
                       width=0.8)
                ax.bar(i, fb, bottom=fr + fi, color=LCOL["below"], width=0.8)
            if r == 0:
                ax.set_title(FLAB[fam])
            if r == 1:
                ax.set_xticks(range(len(epss)))
                ax.set_xticklabels([f"{e:g}" for e in epss], fontsize=8.5)
                ax.set_xlabel(r"$\varepsilon$")
            if c == 0:
                ax.set_ylabel(f"{cond}-trained\nshare (τ={tau:g})")
            ax.grid(False)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=LCOL[k])
                for k in ("robust", "inconclusive", "below")],
               ["certified robust", "inconclusive", "certified below τ"],
               loc="outside lower center", ncol=3)
    fig.suptitle("What the certificate says, per cell", fontsize=12)
    return fig


def fig_a6_scatter(cert, family="vogn", eps=0.05, tau=0.9):
    """A6: one raw cell, three-way colored."""
    g = cert[(cert.family == family) & (cert.condition == "robust")
             & (cert.eps == eps)]
    d = al.add_labels(g, tau)
    rng = np.random.default_rng(0)
    yj = d["p_safe_point"] + rng.uniform(-0.015, 0.015, len(d))
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for lab in ("robust", "inconclusive", "below"):
        m = d["cert_label"] == lab
        ax.scatter(d.loc[m, al.MI], yj[m], s=18, color=LCOL[lab],
                   alpha=0.8, lw=0, label=f"{lab} (n={int(m.sum())})")
    ax.set_xlabel("mutual information (clean input)")
    ax.set_ylabel(r"$\hat P_{\mathrm{safe}}$ (pessimistic, jittered)")
    ax.set_title(rf"{FLAB[family]}, robust, $\varepsilon$={eps:g} "
                 "(5 seeds pooled)")
    ax.legend(loc="center right", fontsize=9)
    return fig