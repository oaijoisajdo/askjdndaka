"""
Across-seed aggregation and matched robust-minus-standard effects.

The seed is the replicate. Everything here consumes seed-level statistics and
never touches an input row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import metrics as M
from analysis.loading import RUN_KEYS, dedupe_standard

# Metrics where a DECREASE is an improvement. The plan asks for paired effects
# signed so positive always means "robust training helped", which means these
# are differenced standard - robust and everything else robust - standard.
LOWER_IS_BETTER = {
    "nll", "brier", "ece", "nll_auc", "brier_auc", "ece_auc",
    "subset_ece_clean", "subset_ece_adv",
    "subset_misclassification_adv", "attack_success_rate",
    "attack_success_auc", "mean_unknown_frac", "mean_interval_width",
    "mean_ci_width", "selective_risk",
}

_ID_COLUMNS = {*RUN_KEYS, "seed", "eval_split", "run_file", "n_inputs",
               "n_used", "n_clean_correct", "n_conclusive", "n_inconclusive",
               "n_retained", "score_constant"}


def _value_columns(df: pd.DataFrame, extra_keys) -> list[str]:
    """
    Numeric metric columns only.

    Booleans are excluded explicitly: pandas reports bool as a numeric dtype,
    so a diagnostic flag would otherwise be averaged into a summary row and,
    worse, differenced in ``paired_effects`` where numpy refuses to subtract
    bools at all.
    """
    skip = _ID_COLUMNS | set(extra_keys)
    return [
        c for c in df.columns
        if c not in skip
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]


def summarize(df: pd.DataFrame, extra_keys=("eps",)) -> pd.DataFrame:
    """
    Mean, SD, seed SE, 95% t interval and n_valid for every numeric column.

    Standard rows are deduplicated first: the same standard model appears once
    per robust point in the JSONs, and counting it repeatedly would understate
    the seed SE without changing the mean.

    Output is long (one row per metric) so that a metric which exists for only
    some families does not create a ragged wide table.
    """
    if df.empty:
        return pd.DataFrame()

    df = dedupe_standard(df, extra_keys=extra_keys)
    group = ["family", "condition", "train_eps", "rob_lam", *extra_keys]
    group = [c for c in group if c in df.columns]
    values = _value_columns(df, extra_keys)

    rows = []
    for keys, g in df.groupby(group, dropna=False):
        base = dict(zip(group, keys if isinstance(keys, tuple) else (keys,)))
        base["n_seeds_present"] = g["seed"].nunique()
        for column in values:
            rows.append({**base, "metric": column,
                         **M.summarize_across(g[column].to_numpy())})
    return pd.DataFrame(rows)


def paired_effects(df: pd.DataFrame, extra_keys=("eps",)) -> pd.DataFrame:
    """
    Matched robust-minus-standard differences, within seed.

    Pairing is on (family, seed, *extra_keys): the standard block in a given
    file was trained with the same seed as the robust block beside it, so the
    difference removes the seed's contribution to variance -- which is large
    relative to the training effect and is the whole reason the plan insists
    on matched seeds.

    Signs are flipped for LOWER_IS_BETTER metrics so a positive effect always
    means robust training improved the outcome. The convention is recorded per
    row in ``direction`` so nothing downstream has to re-derive it.
    """
    if df.empty:
        return pd.DataFrame()

    keys = ["family", "seed", *extra_keys]
    keys = [c for c in keys if c in df.columns]
    values = _value_columns(df, extra_keys)

    standard = (df[df["condition"] == "standard"]
                .drop_duplicates(subset=keys)
                .set_index(keys))
    robust = df[df["condition"] == "robust"]

    rows = []
    for _, r in robust.iterrows():
        index = tuple(r[k] for k in keys)
        if index not in standard.index:
            continue
        s = standard.loc[index]
        if isinstance(s, pd.DataFrame):          # defensive: duplicate standard
            s = s.iloc[0]
        for column in values:
            s_val, r_val = s.get(column, np.nan), r[column]
            if not (np.isfinite(s_val) and np.isfinite(r_val)):
                effect = np.nan
            elif column in LOWER_IS_BETTER:
                effect = s_val - r_val
            else:
                effect = r_val - s_val
            rows.append({
                **{k: r[k] for k in keys},
                "train_eps": r["train_eps"], "rob_lam": r["rob_lam"],
                "metric": column,
                "direction": "standard-robust" if column in LOWER_IS_BETTER
                else "robust-standard",
                "standard": s_val, "robust": r_val, "effect": effect,
            })
    return pd.DataFrame(rows)


def summarize_effects(effects: pd.DataFrame, extra_keys=("eps",)) -> pd.DataFrame:
    """Across-seed summary of the paired effects, same interval convention."""
    if effects.empty:
        return pd.DataFrame()
    group = ["family", "train_eps", "rob_lam", *extra_keys, "metric", "direction"]
    group = [c for c in group if c in effects.columns]
    rows = []
    for keys, g in effects.groupby(group, dropna=False):
        rows.append({
            **dict(zip(group, keys if isinstance(keys, tuple) else (keys,))),
            **M.summarize_across(g["effect"].to_numpy()),
        })
    return pd.DataFrame(rows)
