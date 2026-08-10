from __future__ import annotations
import json
from pathlib import Path
import jax
import flax.nnx as nnx
import orbax.checkpoint as ocp

import hashlib
from typing import Any, Callable

from models import VI_BNN, PriorConfig, MC_BNN, BBBLinear, VOGNLinear

PROJECT_ROOT = Path(__file__).parent.parent.parent  # training -> src -> project root
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def save_model(model, path: str | Path, *, config: dict | None = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    cfg = model.get_config() if config is None else config
    (path / "config.json").write_text(json.dumps(to_jsonable(cfg), indent=2, sort_keys=True))

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path / "state", nnx.state(model), force=True)
    checkpointer.wait_until_finished()


def load_model(path: str | Path, *, seed: int | None = None):

    path = Path(path)

    full_cfg = json.loads((path / "config.json").read_text())

    if "model" in full_cfg:
        model_cfg = dict(full_cfg["model"])

        if seed is None:
            seed = full_cfg.get("training", {}).get("seed", 0)
    else:
        model_cfg = dict(full_cfg)

        if seed is None:
            seed = 0

    model_type = model_cfg.pop("model_type")

    if model_type == "bbb":
        prior_cfg = PriorConfig(**model_cfg.pop("prior"))

        layer_kwargs = {
            "rho_init": model_cfg.pop("rho_init", -3.0),
            "init": model_cfg.pop("init", "xavier"),
        }

        model = nnx.eval_shape(
            lambda: VI_BNN(
                **model_cfg,
                prior=prior_cfg,
                layer_cls=BBBLinear,
                layer_kwargs=layer_kwargs,
                rngs=nnx.Rngs(seed),
            )
        )

    elif model_type == "vogn":
        prior_cfg = PriorConfig(**model_cfg.pop("prior"))

        layer_kwargs = {
            "dataset_size": model_cfg.pop("dataset_size"),
            "s_init": model_cfg.pop("s_init", 1.0),
            "init": model_cfg.pop("init", "xavier"),
        }

        model = nnx.eval_shape(
            lambda: VI_BNN(
                **model_cfg,
                prior=prior_cfg,
                layer_cls=VOGNLinear,
                layer_kwargs=layer_kwargs,
                rngs=nnx.Rngs(seed),
            )
        )

    elif model_type in ("mc_dropout", "deterministic"):
        model = nnx.eval_shape(
            lambda: MC_BNN(
                **model_cfg,
                rngs=nnx.Rngs(seed),
            )
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    graphdef, abstract_state = nnx.split(model)
    abstract_state = _pin_to_local_device(abstract_state)
    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(path / "state", abstract_state)

    return nnx.merge(graphdef, restored_state)

def _pin_to_local_device(abstract_state):
    """Give every abstract leaf an explicit sharding for the *current* devices.

    nnx.split(nnx.eval_shape(...)) yields ShapeDtypeStructs without sharding,
    so Orbax falls back to the sharding stored in the checkpoint metadata. A
    checkpoint written on GPU then refuses to restore on CPU ("Topology
    mismatch"). Supplying the target topology explicitly makes restore
    device-agnostic in both directions.
    """
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def fix(leaf):
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding)
        return leaf

    return jax.tree.map(fix, abstract_state)
def to_jsonable(x: Any) -> Any:
    """
    Convert dataclasses / Paths / tuples into stable JSON-compatible objects.
    """
    if hasattr(x, "__dataclass_fields__"):
        return {
            k: to_jsonable(getattr(x, k))
            for k in x.__dataclass_fields__
        }

    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in sorted(x.items())}

    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    if isinstance(x, Path):
        return str(x)

    return x


def canonical_json(cfg: dict) -> str:
    return json.dumps(
        to_jsonable(cfg),
        sort_keys=True,
        indent=None,
        separators=(",", ":"),
    )


def config_hash(cfg: dict, n_chars: int = 10) -> str:
    s = canonical_json(cfg)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n_chars]


def save_config(cfg: dict, path: str | Path) -> None:
    if path is None:
        print("No config path provided")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(
            to_jsonable(cfg),
            f,
            indent=2,
            sort_keys=True,
        )


def load_config(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        return json.load(f)


def configs_match(cfg_a: dict, cfg_b: dict) -> bool:
    return canonical_json(cfg_a) == canonical_json(cfg_b)

def build_checkpoint_config(
    model,
    *,
    seed: int,
    lr: float,
    train_steps: int,
    beta: float,
    robust_train: bool = False,
    epsilon: float = 0.0,
    rob_lam: float = 1.0,
    data_config=None,
    robust_schedule: dict | None = None,
    vogn_beta1: float | None = None,
    vogn_beta2: float | None = None,
    vogn_eps: float | None = None,
) -> dict:
    robust_train = bool(robust_train)
    model_type = model.model_type

    if robust_train:
        loss_cfg = {
            "name": "wicker_robust_likelihood",
            "version": 1,
            "robust_train": True,
            "epsilon": float(epsilon),
            "rob_lam": float(rob_lam),
        }

        if robust_schedule is not None:
            loss_cfg["schedule"] = robust_schedule
    else:
        loss_cfg = {
            "name": "standard_likelihood",
            "version": 1,
            "robust_train": False,
            "epsilon": 0.0,
            "rob_lam": 1.0,
        }

    training_cfg = {
        "seed": seed,
        "lr": lr,
        "train_steps": train_steps,
        "loss": loss_cfg,
    }

    if model_type == "vogn":
        training_cfg.update({
            "optimizer": "vogn",
            "vogn_beta1": 0.9 if vogn_beta1 is None else float(vogn_beta1),
            "vogn_beta2": 0.999 if vogn_beta2 is None else float(vogn_beta2),
            "vogn_eps": 1e-8 if vogn_eps is None else float(vogn_eps),
        })
    else:
        training_cfg.update({
            "optimizer": "adam",
            "beta": beta,
        })

    cfg = {
        "model": model.get_config(),
        "training": training_cfg,
    }

    if data_config is not None:
        cfg["data"] = (
            data_config.__dict__
            if hasattr(data_config, "__dict__")
            else data_config
        )

    return cfg


def _float_token(x: float) -> str:
    return (
        f"{float(x):g}"
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )


def checkpoint_name_from_config(cfg: dict) -> str:
    model_type = cfg["model"]["model_type"]
    loss_cfg = cfg.get("training", {}).get("loss", {})

    if loss_cfg.get("robust_train", False):
        tag = (
            "robust"
            f"_eps{_float_token(loss_cfg['epsilon'])}"
            f"_lam{_float_token(loss_cfg['rob_lam'])}"
        )
    else:
        tag = "standard"

    h = config_hash(cfg)
    return f"{model_type}_{tag}_{h}"

def load_or_train(
    model,
    train_loader,
    train_fn: Callable,
    *,
    val_loader=None,
    seed=0,
    lr: float = 1e-3,
    train_steps: int = 1500,
    beta: float = 1.0,
    force_retrain: bool = False,
    data_config=None,
    ckpt_dir = CHECKPOINTS_DIR,
    **train_kwargs,
):
    robust_train = bool(train_kwargs.get("robust_train", False))
    epsilon = float(train_kwargs.get("epsilon", 0.0))
    rob_lam = float(train_kwargs.get("rob_lam", 1.0))
    epsilon_warmup_steps = int(train_kwargs.get("epsilon_warmup_steps", 0))
    if robust_train and epsilon_warmup_steps == 0:
            print(
                f"WARNING: robust_train=True, epsilon={epsilon}, but "
                f"epsilon_warmup_steps=0"
            )

    robust_schedule = {
        k: train_kwargs[k]
        for k in (
            "epsilon_schedule",
            "rob_lam_schedule",
            "epsilon_warmup_steps",)
        if k in train_kwargs
    } or None

    try:
        vogn_beta1 = float(train_kwargs.get("vogn_beta1", 0.9))
        vogn_beta2 = float(train_kwargs.get("vogn_beta2", 0.999))
        vogn_eps = float(train_kwargs.get("vogn_eps", 1e-8))
    except None:
        vogn_beta1 = None
        vogn_beta2 = None
        vogn_eps = None

    ckpt_cfg = build_checkpoint_config(
        model,
        seed=seed,
        lr=lr,
        train_steps=train_steps,
        beta=beta,
        robust_train=robust_train,
        epsilon=epsilon,
        rob_lam=rob_lam,
        robust_schedule=robust_schedule,
        data_config=data_config,
        vogn_beta1=vogn_beta1,
        vogn_beta2=vogn_beta2,
        vogn_eps=vogn_eps,
    )

    name = checkpoint_name_from_config(ckpt_cfg)
    if ckpt_dir is not None:
        model_path = ckpt_dir / name
        config_path = model_path / "config.json"

        if model_path.exists() and config_path.exists() and not force_retrain:
            saved_cfg = load_config(config_path)

            if configs_match(saved_cfg, ckpt_cfg):
                print(f"Loading checkpoint {name}")
                return load_model(model_path)

            print(f"Checkpoint config mismatch for {name}; retraining.")

        elif model_path.exists() and not config_path.exists():
            print(f"Checkpoint {name} has no config.json; retraining for safety.")

        else:
            print(f"No checkpoint for config {name}")
    else:
        model_path = None
        config_path = None
    trained_model = train_fn(
        model,
        train_loader,
        val_loader=val_loader,
        save_dir=model_path,
        seed=seed,
        lr=lr,
        train_steps=train_steps,
        beta=beta,
        **train_kwargs,
    )

    save_config(ckpt_cfg, config_path)

    return trained_model