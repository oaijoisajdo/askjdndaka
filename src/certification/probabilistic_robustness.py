"""
Statistical estimation of BNN probabilistic robustness.

Algorithm 1 of Cardelli, Kwiatkowska, Laurenti, Paoletti, Patane, Wicker,
"Statistical Guarantees for the Robustness of Bayesian Neural Networks",
IJCAI 2019 (arXiv:1903.01980), sequential Massart scheme of Jegourel et
al. (2018).

Paper -> code:
    p_j = P(phi_j(f^w) | D)   violation probability (Problems 1, 2)
    theta, gamma              error bound and failure probability, Eqn (2)
    alpha < gamma             confidence of the running interval I_p = [a, b]
    Eqn (3), Eqn (4)          chernoff_sample_bound, massart_sample_bound
    Algorithm 1               estimate_probabilistic_robustness
    phi_1, phi_2              make_phi1, make_phi2

Deviation from the paper: the deterministic sub-routine. The paper decides
SAT(phi_j) with the complete reachability method of Ruan et al.; here it is
the sound but *incomplete* interval machinery shared with certify.py, so a
sample can be certified safe, certified violating, or undecided. Eqn (2)
holds for the mean of whatever Bernoulli is actually sampled, so undecided
samples are resolved by a fixed policy that fixes the bound direction:

    unknown_as="violation" -> p_j <= p_hat + theta   (default)
    unknown_as="safe"      -> p_j >= p_hat - theta

Running both sandwiches p_j; the gap is the incompleteness of the verifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple
from collections.abc import Sequence, Callable
from functools import partial                  # add

import jax
import jax.numpy as jnp
from scipy.stats import beta

from .bound_propagation import (
    forward_layers,
    propagate_deterministic_IBP,
    propagate_deterministic_LBP,
    propagate_IBP,
    propagate_LBP,
    _propagate_deterministic_ibp_kernel,      # add
    _propagate_deterministic_lbp_kernel,      # add
)
from .certify import pred_safe, pred_unsafe
from .posteriors import Posterior, make_deterministic_box_from_layers

Array = jax.Array
Weights = list[tuple[Array, Array]]          # sampled layers [(w, b), ...]
ClassSelection = Literal["sample", "true_label", "argmax"]
UnknownPolicy = Literal["violation", "safe"]
CandidatesFn = Callable[[Weights, Array, int | None], Sequence[Array]]
Verdict = Literal["safe", "unsafe", "unknown"]
PropertyFn = Callable[[Weights, Array], Verdict]  # (layers, key) -> phi holds?

_SAFE, _UNSAFE, _UNKNOWN = 0, 1, 2
_VERDICTS: tuple[Verdict, ...] = ("safe", "unsafe", "unknown")

def _decode_verdict(code: Array) -> Verdict:
    """The single intentional device-to-host transfer per verdict."""
    return _VERDICTS[int(jax.device_get(code))]

@dataclass
class RobustnessEstimate:
    p_hat: float          # Eqn (1)
    n: int                # samples drawn
    k: int                # SAT count
    unk: int
    ci_low: float         # final 1 - alpha interval
    ci_high: float
    theta: float
    gamma: float
    alpha: float




@dataclass
class RobustnessBoundsEstimate:
    """
    Two-sided result obtained from one shared stream of verifier verdicts.

    ``unknown_as_safe`` counts only certified violations and gives the lower
    side of the true violation probability. ``unknown_as_violation`` counts
    violations and unknowns and gives the upper side.

    Each side individually has failure probability at most ``gamma``. Without
    an additional correction, the simultaneous bracket has failure probability
    at most ``2 * gamma`` by the union bound.
    """

    unknown_as_safe: RobustnessEstimate
    unknown_as_violation: RobustnessEstimate
    n_property_evaluations: int

    @property
    def violation_lower(self) -> float:
        estimate = self.unknown_as_safe
        return max(0.0, estimate.p_hat - estimate.theta)

    @property
    def violation_upper(self) -> float:
        estimate = self.unknown_as_violation
        return min(1.0, estimate.p_hat + estimate.theta)

    @property
    def robustness_lower(self) -> float:
        return 1.0 - self.violation_upper

    @property
    def robustness_upper(self) -> float:
        return 1.0 - self.violation_lower


@dataclass
class _EstimatorState:
    unknown_as: UnknownPolicy
    n_chernoff: int
    n_max: int
    n: int = 0
    k: int = 0
    unk: int = 0
    done: bool = False


def _new_estimator_state(
    *,
    theta: float,
    gamma: float,
    unknown_as: UnknownPolicy,
) -> _EstimatorState:
    if unknown_as not in ("violation", "safe"):
        raise ValueError(f"Unknown unknown_as policy: {unknown_as!r}.")
    n_chernoff = chernoff_sample_bound(theta, gamma)
    return _EstimatorState(
        unknown_as=unknown_as,
        n_chernoff=n_chernoff,
        n_max=n_chernoff,
    )


def _update_estimator_state(
    state: _EstimatorState,
    verdict: Verdict,
    *,
    theta: float,
    gamma: float,
    alpha: float,
) -> None:
    if state.done:
        return
    if verdict not in ("safe", "unsafe", "unknown"):
        raise ValueError(f"Invalid property verdict: {verdict!r}.")

    state.unk += int(verdict == "unknown")
    state.k += int(
        verdict == "unsafe"
        or (verdict == "unknown" and state.unknown_as == "violation")
    )
    state.n += 1

    a, b = clopper_pearson_ci(k=state.k, n=state.n, alpha=alpha)
    n_massart = massart_sample_bound(
        theta=theta,
        gamma=gamma,
        alpha=alpha,
        a=a,
        b=b,
    )
    state.n_max = min(n_massart, state.n_chernoff)
    state.done = state.n >= state.n_max


def _finish_estimator_state(
    state: _EstimatorState,
    *,
    theta: float,
    gamma: float,
    alpha: float,
) -> RobustnessEstimate:
    ci_low, ci_high = clopper_pearson_ci(
        k=state.k,
        n=state.n,
        alpha=alpha,
    )
    return RobustnessEstimate(
        p_hat=state.k / state.n,
        n=state.n,
        k=state.k,
        unk=state.unk,
        ci_low=ci_low,
        ci_high=ci_high,
        theta=theta,
        gamma=gamma,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------

def estimate_probabilistic_robustness(
    *,
    posterior: Posterior,
    property_fn: PropertyFn,
    key: Array,
    theta: float = 0.075,
    gamma: float = 0.075,
    alpha: float = 0.05,
    unknown_as: UnknownPolicy = "violation",
) -> RobustnessEstimate:
    """
    Estimate p = P_w(property_fn(f^w) is True) with P(|p_hat - p| > theta)
    <= gamma. Paper defaults (Section 6.1) give n^C = 292 worst case.

    f^w is robust/safe with probability at least 1 - eta iff
    p_hat + theta <= eta, with probability >= 1 - gamma.
    """
    state = _new_estimator_state(
        theta=theta,
        gamma=gamma,
        unknown_as=unknown_as,
    )

    while not state.done:                               # line 3
        key, key_w, key_property = jax.random.split(key, 3)

        layers = posterior.sample(key_w)               # line 4
        # line 5 (class sampling) happens inside phi_2; line 6:
        verdict = property_fn(layers, key_property)     # lines 4-6
        _update_estimator_state(                        # lines 7-11
            state,
            verdict,
            theta=theta,
            gamma=gamma,
            alpha=alpha,
        )

    return _finish_estimator_state(                    # line 13
        state,
        theta=theta,
        gamma=gamma,
        alpha=alpha,
    )


def estimate_probabilistic_robustness_bounds(
    *,
    posterior: Posterior,
    property_fn: PropertyFn,
    key: Array,
    theta: float = 0.075,
    gamma: float = 0.075,
    alpha: float = 0.05,
) -> RobustnessBoundsEstimate:
    """
    Run both unknown policies while evaluating each sampled network once.

    A side freezes as soon as its own sequential Massart stopping rule is met;
    the shared stream continues only until the other side also stops. Thus the
    property is evaluated ``max(n_safe, n_violation)`` times rather than
    ``n_safe + n_violation`` times.
    """
    safe_state = _new_estimator_state(
        theta=theta,
        gamma=gamma,
        unknown_as="safe",
    )
    violation_state = _new_estimator_state(
        theta=theta,
        gamma=gamma,
        unknown_as="violation",
    )
    n_property_evaluations = 0

    while not (safe_state.done and violation_state.done):
        key, key_w, key_property = jax.random.split(key, 3)
        layers = posterior.sample(key_w)
        verdict = property_fn(layers, key_property)
        n_property_evaluations += 1

        _update_estimator_state(
            safe_state,
            verdict,
            theta=theta,
            gamma=gamma,
            alpha=alpha,
        )
        _update_estimator_state(
            violation_state,
            verdict,
            theta=theta,
            gamma=gamma,
            alpha=alpha,
        )

    return RobustnessBoundsEstimate(
        unknown_as_safe=_finish_estimator_state(
            safe_state,
            theta=theta,
            gamma=gamma,
            alpha=alpha,
        ),
        unknown_as_violation=_finish_estimator_state(
            violation_state,
            theta=theta,
            gamma=gamma,
            alpha=alpha,
        ),
        n_property_evaluations=n_property_evaluations,
    )




# ---------------------------------------------------------------------------
# Sample-size bounds and confidence interval
# ---------------------------------------------------------------------------

def _check_unit_interval(name: str, x: float) -> None:
    if not (0.0 < x < 1.0):
        raise ValueError(f"{name} must be in (0, 1).")


def _strict_integer_bound(x: float) -> int:
    """Smallest integer strictly greater than x (the 'n > ...' bounds)."""
    return math.floor(x) + 1


def chernoff_sample_bound(theta: float, gamma: float) -> int:
    """Eqn (3): smallest n with n > 1 / (2 theta^2) * log(2 / gamma)."""
    _check_unit_interval("theta", theta)
    _check_unit_interval("gamma", gamma)
    return _strict_integer_bound(
        math.log(2.0 / gamma) / (2.0 * theta * theta)
    )


def clopper_pearson_ci(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Exact binomial interval with confidence 1 - alpha (line 9)."""
    if not (0 <= k <= n):
        raise ValueError("Need 0 <= k <= n.")
    _check_unit_interval("alpha", alpha)
    if n == 0:
        return 0.0, 1.0

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def massart_sample_bound(
    *, theta: float, gamma: float, alpha: float, a: float, b: float
) -> int:
    """
    Eqn (4), Massart bound evaluated on I_p = [a, b]: smallest n with

        n > 2 / (9 theta^2) * log(2 / (gamma - alpha)) * F,
        F = (3b + theta)(3(1 - b) - theta)  if b < 1/2
            (3(1 - a) + theta)(3a + theta)  if a > 1/2
            (3/2 + theta)^2                 otherwise.
    """
    _check_unit_interval("theta", theta)
    if not (0.0 < alpha < gamma < 1.0):
        raise ValueError("Need 0 < alpha < gamma < 1.")

    if b < 0.5:
        factor = (3.0 * b + theta) * (3.0 * (1.0 - b) - theta)
    elif a > 0.5:
        factor = (3.0 * (1.0 - a) + theta) * (3.0 * a + theta)
    else:
        factor = (1.5 + theta) ** 2

    prefactor = (2.0 / (9.0 * theta * theta)) * math.log(2.0 / (gamma - alpha))
    return _strict_integer_bound(prefactor * factor)

# ---------------------------------------------------------------------------
# Shared pieces for the property functions
# ---------------------------------------------------------------------------


def _uses_deterministic_ibp(propagate_fn: Callable) -> bool:
    return propagate_fn in (propagate_IBP, propagate_deterministic_IBP)


def _propagate_sampled_layers(
    layers: Weights,
    x_L: Array,
    x_U: Array,
    *,
    propagate_fn: Callable,
) -> tuple[Array, Array]:
    """
    Route known propagation engines to their fixed-weight implementations.

    This preserves the old ``propagate_fn=propagate_IBP/propagate_LBP`` API.
    Unknown custom functions retain the legacy ``(WeightBox, x_L, x_U)``
    contract.
    """
    if _uses_deterministic_ibp(propagate_fn):
        return propagate_deterministic_IBP(layers, x_L, x_U)
    if propagate_fn in (propagate_LBP, propagate_deterministic_LBP):
        return propagate_deterministic_LBP(layers, x_L, x_U)

    box = make_deterministic_box_from_layers(layers)
    return propagate_fn(box, x_L, x_U)


def _intersect_bounds(
    first: tuple[Array, Array],
    second: tuple[Array, Array],
) -> tuple[Array, Array]:
    return (
        jnp.maximum(first[0], second[0]),
        jnp.minimum(first[1], second[1]),
    )


def point_logits(
    layers: Weights,
    x: Array,
    *,
    propagate_fn=None,
) -> Array:
    """
    Exact logits of one sampled net at a point via a direct compiled forward.

    ``propagate_fn`` is accepted for source compatibility but deliberately
    ignored: a concrete-point evaluation needs no bound propagation.
    """
    del propagate_fn
    return jnp.ravel(forward_layers(layers, x))


def softmax_interval_bounds(
    logits_L: Array, logits_U: Array
) -> tuple[Array, Array]:
    """
    Sound elementwise softmax bounds over a logit box, from monotonicity
    of softmax in its own logit (up) and in the others (down):

        sigma_h >= sigmoid(l_h - LSE_{j != h} u_j)
        sigma_h <= sigmoid(u_h - LSE_{j != h} l_j).
    """
    def lse_excl(v: Array) -> Array:
        C = v.shape[-1]
        rows = jnp.broadcast_to(v, (C, C))
        return jax.scipy.special.logsumexp(
            jnp.where(jnp.eye(C, dtype=bool), -jnp.inf, rows), axis=-1
        )

    logits_L, logits_U = jnp.ravel(logits_L), jnp.ravel(logits_U)
    return (
        jax.nn.sigmoid(logits_L - lse_excl(logits_U)),
        jax.nn.sigmoid(logits_U - lse_excl(logits_L)),
    )


def _corners(x_L: Array, x_U: Array) -> Sequence[Array]:
    return (x_L, x_U)


# ---------------------------------------------------------------------------
# Problem 1: phi_1 = exists x in T. |sigma(f^w(x*)) - sigma(f^w(x))|_p > delta
# ---------------------------------------------------------------------------


def make_phi1(*, x_star, x_L, x_U, delta, p_norm=jnp.inf,
              candidates_fn=None, propagate_fn=propagate_IBP,
              ibp_first=False) -> PropertyFn:
    if delta < 0.0:
        raise ValueError("delta must be non-negative.")

    x_star = jnp.asarray(x_star).reshape(-1)
    x_l = jnp.asarray(x_L).reshape(-1)
    x_u = jnp.asarray(x_U).reshape(-1)
    if x_star.shape != x_l.shape or x_l.shape != x_u.shape:
        raise ValueError("x_star, x_L, x_U must have equal flattened shapes.")
    if bool(jnp.any(x_l > x_u)):                        # validated once, here
        raise ValueError("Every input lower bound must be <= its upper bound.")

    engine = ("ibp" if _uses_deterministic_ibp(propagate_fn)
              else "lbp" if propagate_fn in (propagate_LBP,
                                             propagate_deterministic_LBP)
              else None)
    fuseable = candidates_fn is None or isinstance(candidates_fn, PGDConfig)

    if engine is not None and fuseable:
        delta_arr = jnp.asarray(delta)
        p_norm_static = float(p_norm)                   # hashable static

        def phi1(layers: Weights, key: Array) -> Verdict:
            return _decode_verdict(_phi1_verdict_code(
                layers, key, x_star, x_l, x_u, delta_arr,
                engine=engine, ibp_first=ibp_first,
                p_norm=p_norm_static, pgd=candidates_fn))
        return phi1

    else:
    # ... existing phi1 body stays below, unchanged, as the fallback for
    # custom propagate_fn callables / custom candidates_fn ...

        def norm(v: Array) -> float:
            return float(jnp.linalg.norm(v, ord=p_norm))

        def bounds_are_safe(
            bounds: tuple[Array, Array],
            sigma_star: Array,
        ) -> bool:
            sig_lo, sig_hi = softmax_interval_bounds(*bounds)
            worst = jnp.maximum(sigma_star - sig_lo, sig_hi - sigma_star)
            return norm(jnp.clip(worst, 0.0)) <= delta

        def phi1(layers: Weights, key: Array) -> Verdict:
            sigma_star = jax.nn.softmax(point_logits(layers, x_star))
            ibp_bounds = None
            if ibp_first and not _uses_deterministic_ibp(propagate_fn):
                ibp_bounds = propagate_deterministic_IBP(layers, x_L, x_U)
                if bounds_are_safe(ibp_bounds, sigma_star):
                    return "safe"

            bounds = _propagate_sampled_layers(
                layers,
                x_L,
                x_U,
                propagate_fn=propagate_fn,
            )
            if ibp_bounds is not None:
                bounds = _intersect_bounds(bounds, ibp_bounds)
            safe = bounds_are_safe(bounds, sigma_star)

            unsafe = False
            if not safe:
                points = (
                    _corners(x_L, x_U)
                    if candidates_fn is None
                    else candidates_fn(layers, key, None,)
                )
                points = tuple(jnp.clip(x, x_L, x_U) for x in points)
                unsafe = any(
                    norm(
                        sigma_star
                        - jax.nn.softmax(point_logits(layers, x))
                    )
                    > delta
                    for x in points
                )

            if unsafe:
                return "unsafe"
            if safe:
                return "safe"
            return "unknown"

        return phi1

@partial(jax.jit, static_argnames=("engine", "ibp_first", "p_norm", "pgd"))
def _phi1_verdict_code(layers, key, x_star, x_l, x_u, delta, *,
                       engine, ibp_first, p_norm, pgd):
    sigma_star = jax.nn.softmax(point_logits(layers, x_star))

    def bounds_safe(bounds):
        sig_lo, sig_hi = softmax_interval_bounds(*bounds)
        worst = jnp.maximum(sigma_star - sig_lo, sig_hi - sigma_star)
        return jnp.linalg.norm(jnp.clip(worst, 0.0), ord=p_norm) <= delta

    def test_candidates(_):
        if pgd is None:
            points = jnp.stack(_corners(x_l, x_u))
        else:
            def loss(x):
                sigma = jax.nn.softmax(point_logits(layers, x))
                return jnp.linalg.norm(sigma - sigma_star, ord=p_norm)
            points = _pgd_points(layers, key, x_star, x_l, x_u, pgd, loss)
        sigmas = jax.nn.softmax(
            jax.vmap(lambda x: point_logits(layers, x))(points), axis=-1)
        deviations = jnp.linalg.norm(
            sigmas - sigma_star[None, :], ord=p_norm, axis=-1)
        return jnp.where(jnp.any(deviations > delta),
                         _UNSAFE, _UNKNOWN).astype(jnp.int8)

    def decide(bounds):
        return jax.lax.cond(bounds_safe(bounds),
                            lambda _: jnp.asarray(_SAFE, jnp.int8),
                            test_candidates, operand=None)

    ibp_bounds = (_propagate_deterministic_ibp_kernel(layers, x_l, x_u)
                  if engine == "ibp" or ibp_first else None)
    if engine == "ibp":
        return decide(ibp_bounds)

    def lbp_rung(_):
        bounds = _propagate_deterministic_lbp_kernel(
            layers, x_l, x_u, relu_lower="adaptive", tighten_with_ibp=False)
        if ibp_bounds is not None:
            bounds = _intersect_bounds(bounds, ibp_bounds)
        return decide(bounds)

    if ibp_first:
        return jax.lax.cond(bounds_safe(ibp_bounds),
                            lambda _: jnp.asarray(_SAFE, jnp.int8),
                            lbp_rung, operand=None)
    return lbp_rung(None)

# ---------------------------------------------------------------------------
# Problem 2: phi_2 = exists x in T. m(x*) != m(x)
# ---------------------------------------------------------------------------

def make_phi2(
    *,
    x_star: Array,
    x_L: Array,
    x_U: Array,
    y_dim: int,
    class_selection: ClassSelection = "sample",
    true_label: int | None = None,
    candidates_fn: CandidatesFn | None = None,
    propagate_fn=propagate_IBP,
    ibp_first: bool = False,
) -> PropertyFn:
    x_star = jnp.asarray(x_star).reshape(-1)
    x_l = jnp.asarray(x_L).reshape(-1)
    x_u = jnp.asarray(x_U).reshape(-1)
    if bool(jnp.any(x_l > x_u)):                       # validated ONCE, here
        raise ValueError("Every input lower bound must be <= its upper bound.")

    engine = ("ibp" if _uses_deterministic_ibp(propagate_fn)
              else "lbp" if propagate_fn in (propagate_LBP,
                                             propagate_deterministic_LBP)
              else None)
    fuseable = candidates_fn is None or isinstance(candidates_fn, PGDConfig)

    if engine is not None and fuseable:
        label = jnp.asarray(0 if true_label is None else true_label, jnp.int32)

        def phi2(layers: Weights, key: Array) -> Verdict:
            return _decode_verdict(_phi2_verdict_code(
                layers, key, x_star, x_l, x_u, label,
                y_dim=y_dim, class_selection=class_selection,
                engine=engine, ibp_first=ibp_first, pgd=candidates_fn))
        return phi2

@partial(jax.jit, static_argnames=(
    "y_dim", "class_selection", "engine", "ibp_first", "pgd"))
def _phi2_verdict_code(layers, key, x_star, x_l, x_u, true_label, *,
                       y_dim, class_selection, engine, ibp_first, pgd):
    if class_selection == "true_label":
        y = true_label
    else:
        logits = point_logits(layers, x_star)
        y = (jnp.argmax(logits) if class_selection == "argmax"
             else jax.random.categorical(key, logits)).astype(jnp.int32)

    def bounds_code(bounds):
        safe = pred_safe(*bounds, y, y_dim)
        unsafe = pred_unsafe(*bounds, y, y_dim)
        return jnp.where(unsafe, _UNSAFE,
                         jnp.where(safe, _SAFE, _UNKNOWN)).astype(jnp.int8)

    def test_candidates(_):
        if pgd is None:
            points = jnp.stack(_corners(x_l, x_u))
        else:
            def loss(x):
                z = point_logits(layers, x)
                wrong = jnp.where(jnp.arange(y_dim) == y, -jnp.inf, z)
                return jnp.max(wrong) - z[y]
            points = _pgd_points(layers, key, x_star, x_l, x_u, pgd, loss)
        logits = jax.vmap(lambda x: point_logits(layers, x))(points)
        unsafe = jnp.any(jnp.argmax(logits, axis=-1) != y)
        return jnp.where(unsafe, _UNSAFE, _UNKNOWN).astype(jnp.int8)

    def decide(bounds):
        code = bounds_code(bounds)
        return jax.lax.cond(code == _UNKNOWN, test_candidates,
                            lambda _: code, operand=None)

    ibp_bounds = (_propagate_deterministic_ibp_kernel(layers, x_l, x_u)
                  if engine == "ibp" or ibp_first else None)
    if engine == "ibp":
        return decide(ibp_bounds)

    def lbp_rung(_):
        bounds = _propagate_deterministic_lbp_kernel(
            layers, x_l, x_u, relu_lower="adaptive", tighten_with_ibp=False)
        if ibp_bounds is not None:
            bounds = _intersect_bounds(bounds, ibp_bounds)
        return decide(bounds)

    if ibp_first:
        ibp_code = bounds_code(ibp_bounds)
        return jax.lax.cond(ibp_code != _UNKNOWN,
                            lambda _: ibp_code, lbp_rung, operand=None)
    return lbp_rung(None)


class PGDConfig(NamedTuple):
    steps: int = 10
    restarts: int = 1
    step_frac: float = 0.25
    p_norm: float = float("inf")

def make_pgd_candidates(*, p_norm=float("inf"), steps=10,
                        step_frac=0.25, restarts=1) -> PGDConfig:
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if restarts < 1:
        raise ValueError("restarts must be positive.")
    if step_frac <= 0.0:
        raise ValueError("step_frac must be positive.")
    return PGDConfig(steps, restarts, step_frac, p_norm)

def _pgd_points(layers, key, x_star, x_l, x_u, cfg, loss):
    step = cfg.step_frac * jnp.max(x_u - x_l)
    grad_loss = jax.grad(loss)

    def one_restart(restart_key):
        x0 = jnp.clip(
            x_star + jax.random.uniform(
                restart_key, x_star.shape, minval=-step, maxval=step),
            x_l, x_u)
        return jax.lax.fori_loop(
            0, cfg.steps,
            lambda _, x: jnp.clip(x + step * jnp.sign(grad_loss(x)), x_l, x_u),
            x0)

    adversarial = jax.vmap(one_restart)(jax.random.split(key, cfg.restarts))
    return jnp.concatenate((adversarial, jnp.stack(_corners(x_l, x_u))))