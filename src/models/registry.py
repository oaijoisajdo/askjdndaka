"""src/models/registry.py

Single place that turns a *declarative* model description into a live model,
and back again.

    build_model(family=..., width=..., ..., n_train=...)   # fresh model
    build_from_config(cfg, rngs=...)                       # inverse of
                                                           # VI_BNN.get_config()

Note on `n_train`: VOGNLinear takes `dataset_size` at construction, so the
factory can only run once the train loader exists.

Note on deterministic: MC_BNN reports model_type from p_drop, so a
deterministic net is literally MC_BNN(p_drop=0.0). The registry keeps
"deterministic" and "mc_dropout" as separate *families* anyway, because they
are separate rows in the zoo and separate entries in the manifest.
"""

from __future__ import annotations
from typing import Any
import flax.nnx as nnx

from .bnn import VI_BNN, BBBLinear, VOGNLinear, PriorConfig
from .mc_dropout import MC_BNN

# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------
# A "family" is a training recipe, not an architecture. vogn and vogn_robust
# build the identical module; they differ only in train() kwargs. Keeping the
# distinction here means the manifest can name a run unambiguously without
# inspecting training flags.

FAMILY_TO_MODEL_TYPE: dict[str, str] = {
    "deterministic": "deterministic",
    "mc_dropout": "mc_dropout",
    "bbb": "bbb",
    "vogn": "vogn",
    "vogn_robust": "vogn",
    "bbb_robust": "bbb",
}

BAYESIAN_LAYERS: dict[str, type] = {
    "bbb": BBBLinear,
    "vogn": VOGNLinear,
}

#: Families whose predictive distribution requires MC sampling over weights.
STOCHASTIC_FAMILIES = frozenset(
    {"mc_dropout", "bbb", "vogn", "vogn_robust", "bbb_robust"}
)

#: Families the sound (Wicker-style) Gaussian certifier supports. mc_dropout
#: goes through prob_veri_dropout instead; deterministic through plain IBP.
GAUSSIAN_POSTERIOR_FAMILIES = frozenset({"bbb", "vogn", "vogn_robust", "bbb_robust"})

DEFAULT_P_DROP = 0.2   # matches MC_BNN's own default


def is_robust(family: str) -> bool:
    return family.endswith("_robust")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_model(
    *,
    family: str,
    d_in: int = 784,
    d_out: int = 10,
    width: int = 32,
    depth: int = 1,
    rngs: nnx.Rngs,
    prior: PriorConfig | None = None,
    n_train: int | None = None,
    # family-specific
    rho_init: float = -3.0,
    s_init: float = 1.0,
    p_drop: float = DEFAULT_P_DROP,
    init: str = "xavier",
):
    """Construct a fresh model for `family`.

    `n_train` is required for VOGN families and ignored otherwise.
    `p_drop` is ignored for everything except mc_dropout (deterministic forces 0).
    """
    if family not in FAMILY_TO_MODEL_TYPE:
        raise ValueError(
            f"Unknown family {family!r}. Known: {sorted(FAMILY_TO_MODEL_TYPE)}"
        )

    model_type = FAMILY_TO_MODEL_TYPE[family]
    prior = prior if prior is not None else PriorConfig()

    if model_type in BAYESIAN_LAYERS:
        layer_cls = BAYESIAN_LAYERS[model_type]

        if model_type == "vogn":
            if n_train is None:
                raise ValueError(
                    "VOGN layers need dataset_size at construction; pass "
                    "n_train=len(train_loader.dataset)."
                )
            if prior.name != "gaussian":
                raise ValueError("VOGN requires a Gaussian prior.")
            layer_kwargs: dict[str, Any] = {
                "dataset_size": n_train, "s_init": s_init, "init": init,
            }
        else:  # bbb
            layer_kwargs = {"rho_init": rho_init, "init": init}

        return VI_BNN(
            d_in=d_in, d_out=d_out, width=width, depth=depth,
            rngs=rngs, prior=prior,
            layer_cls=layer_cls, layer_kwargs=layer_kwargs,
        )

    # --- MC_BNN covers both deterministic (p=0) and mc_dropout ------------
    effective_p = 0.0 if model_type == "deterministic" else p_drop
    if model_type == "mc_dropout" and effective_p <= 0.0:
        raise ValueError(
            "family='mc_dropout' with p_drop=0 builds a model whose "
            "model_type is 'deterministic'. Use family='deterministic' "
            "instead, or set p_drop > 0."
        )

    return MC_BNN(
        d_in=d_in, d_out=d_out, width=width, depth=depth,
        p_drop=effective_p, rngs=rngs,
    )


def build_from_config(cfg: dict, *, rngs: nnx.Rngs):
    """Inverse of `VI_BNN.get_config()` (plus the MC_BNN families).

    `cfg` is the flat dict written to config.json. Extra keys that are not
    constructor arguments are ignored, so it is safe to pass a full run config.

    If checkpointing.load_model already rebuilds a model from a saved config,
    it should call *this* rather than keeping a second reconstruction path --
    two of them will drift.
    """
    model_type = cfg["model_type"]
    prior = (
        PriorConfig(**cfg["prior"])
        if isinstance(cfg.get("prior"), dict)
        else PriorConfig()
    )

    family = {
        "bbb": "bbb", "vogn": "vogn",
        "mc_dropout": "mc_dropout", "deterministic": "deterministic",
    }[model_type]

    return build_model(
        family=family,
        d_in=cfg["d_in"],
        d_out=cfg["d_out"],
        width=cfg["width"],
        depth=cfg["depth"],
        rngs=rngs,
        prior=prior,
        n_train=cfg.get("dataset_size"),
        rho_init=cfg.get("rho_init", -3.0),
        s_init=cfg.get("s_init", 1.0),
        p_drop=cfg.get("p_drop", DEFAULT_P_DROP),
        init=cfg.get("init", "xavier"),
    )




def param_count(model) -> int:
    import jax

    state = nnx.state(model, nnx.Param)
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(state)))