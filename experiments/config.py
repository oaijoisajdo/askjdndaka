"""
Central configuration for the adversarial-training / certification runs.

Everything that was a module-level global in adversarial_run.py lives here as
a frozen dataclass. The full config is serialized into every result JSON via
`asdict(CONFIG)`, so a payload is always self-describing.

If you prefer YAML later: `dataclasses.asdict(CONFIG)` round-trips through
yaml.safe_dump / a small loader with no changes to the run script.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    width: int = 512
    depth: int = 1
    arch_tag: str = "mlp512x1"

    prior_sigma: float = 0.2
    rho_init: float = -5.0
    s_init: float = 0.0
    p_drop: float = 0.4

    families: tuple[str, ...] = ("vogn", 
                                 "bbb",
                                 "mc_dropout",
                                 "deterministic"
                                 )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StandardTrainConfig:
    lr: float = 1e-3
    train_steps: int = 12_000
    eval_every: int = 1_200
    save_best: bool = True
    beta: float = 0.1

    def kwargs(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RobustTrainConfig:
    lr: float = 1e-3
    train_steps: int = 12_000
    eval_every: int = 1_200
    save_best: bool = True
    beta: float = 0.1
    robust_train: bool = True
    # NOTE: `best/` is written but never restored -- load_or_train returns the
    # final-step artifact. So val exerts no selection pressure on the weights,
    # and the ramp/selection interaction is a non-issue. Keep it that way: if
    # you ever start loading `best/`, the clean-val-loss criterion will favour
    # early low-eps checkpoints and silently weaken every robust model.
    epsilon_warmup_steps: int = 11_000

    def kwargs(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RobustSweepPoint:
    epsilon: float
    rob_lam: float


# ---------------------------------------------------------------------------
# Attacks (empirical robustness, full validation set, ensemble predictor)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttackConfig:
    pgd_eps: tuple[float, ...] = (0.01, 0.03, 0.05, 0.07, 0.08, 0.1, 0.15)
    steps: int = 40
    random_start: bool = True
    mode: str = "decision"
    # 1 = headline configuration; >1 only for the one-time
    # restart convergence check (Sec. B1 of the handoff).
    restarts: int = 1
    # Seeds: the attack optimizes against MC draws keyed by `seed`; the
    # subsequent evaluation must NOT reuse those draws, otherwise we score an
    # EOT attack on exactly the samples it optimized against.
    eval_seed_offset: int = 10_000


# ---------------------------------------------------------------------------
# Certification (statistical P_safe probe, balanced subset)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CertConfig:
    # 0.08 == trained radius: the number a reviewer asks for first.
    eps_values: tuple[float, ...] = (0.01, 0.05, 0.07, 0.08, 0.1)
    per_class: int = 5              # 50 inputs total
    ibp_first: bool = True

    # Massart/Chernoff estimator. theta=0.075 caps robustness_lower at 0.925
    # and floors it at 0; use the point estimates (also serialized) for any
    # ranking/correlation analysis -- the theta shift is a constant there.
    theta: float = 0.075
    gamma: float = 0.075
    alpha: float = 0.05

    # Falsifier strength. Weak candidates inflate the unknown fraction, which
    # is where the entire interval width comes from at large eps.
    candidate_pgd_steps: int = 50
    candidate_pgd_restarts: int = 3


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationConfig:
    """
    Which split every reported number comes from.

    `train()` uses the validation split for `save_best` checkpoint selection,
    so reporting on that same split leaks selection into the results: the
    reported accuracy is the max over ~`train_steps / eval_every` checkpoints
    rather than a clean estimate. Report on the held-out third split instead.

    The bias is not symmetric across arms. Standard training's checkpoints
    differ mainly by optimization noise, but under the epsilon ramp the robust
    arm's checkpoints are genuinely different models, so `save_best` has more
    to choose between and the selection bias is larger there -- which lands
    directly on the standard-vs-robust deltas that are the headline claims.
    """
    report_split: str = "test"          # "test" | "val"
    # Fail loudly rather than silently reporting on the selection split.
    require_held_out: bool = True
    # Measure (don't assume) disjointness of report and selection splits.
    check_split_overlap: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "mnist"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    mc_samples: int = 50
    n_classes: int = 10
    # Which split every reported number is computed on.
    #   "val"  -- development. Hyperparameters were chosen while looking at
    #             this split, so treat its numbers as in-sample for any
    #             tuning decision.
    #   "test" -- final. Touch once, for the numbers that go in the thesis.
    # Training always uses val_loader for its progress printout regardless;
    # that is harmless because `best/` is never restored.
    eval_split: str = "test"

    model: ModelConfig = field(default_factory=ModelConfig)
    standard_train: StandardTrainConfig = field(default_factory=StandardTrainConfig)
    robust_train: RobustTrainConfig = field(default_factory=RobustTrainConfig)
    robust_sweep: tuple[RobustSweepPoint, ...] = (
        RobustSweepPoint(epsilon=0.08, rob_lam=0.25),
    )
    attack: AttackConfig = field(default_factory=AttackConfig)
    cert: CertConfig = field(default_factory=CertConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def as_dict(self) -> dict:
        return asdict(self)


CONFIG = ExperimentConfig()
