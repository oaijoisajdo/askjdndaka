"""
wicker_loader.py

Load a VOGN posterior trained with Matthew Wicker's `deepbayes`
(https://github.com/matthewwicker/deepbayes) into our own `VI_BNN`
(with `VOGNLinear` layers) and/or a `Posterior`.

A saved deepbayes posterior directory looks like::

    VOGN_MNIST_Posterior/
        arch.json     # Keras Sequential architecture (units, activations)
        info.pkl      # training config: {'N': 60000, 'classes': 10, ...}
        mean.npy      # object array: [W0, b0, W1, b1, ...]  (posterior means)
        var.npy       # object array: [V0, v0, V1, v1, ...]  ("variances")
        model.h5      # Keras weights (the means again -- we don't need it)

`mean.npy` / `var.npy` are numpy *object* arrays whose elements are pickled
TensorFlow `EagerTensor`s. We decode them without importing TensorFlow, via a
custom unpickler that maps `tensorflow...convert_to_tensor` to `np.asarray`.

--------------------------------------------------------------------------
The variance convention (read this before trusting the numbers)
--------------------------------------------------------------------------
deepbayes' VOGN optimizer (`optimizers/blrvi.py`) keeps an internal Fisher /
precision accumulator ``s = self.posterior_var`` and defines a *matched pair*
of methods:

    save():    saved_var = 1 / (N * s)                 -> written to var.npy
    sample():  std       = 1 / sqrt(N * s)             -> scale of N(mu, std)

Eliminating ``s`` from that pair gives the exact relationship

    std = sqrt(saved_var)

so **the posterior standard deviation is the square root of var.npy**. That is
the default here (`variance_mode="variance"`).

deepbayes also ships a *generic* ``Optimizer.load`` that instead does
``inv_softplus(var)`` and stores it under different attribute names
(``posti_var`` rather than ``posterior_var``), which would imply
``std == var`` directly. That path is not the one VOGN's own ``sample`` uses,
so we treat it as legacy. If you ever need to reproduce that behaviour, pass
``variance_mode="std"``. `"raw_s"` is provided for the (unusual) case of a
file that stored the bare accumulator ``s``.
--------------------------------------------------------------------------

Typical use::

    from training.wicker_loader import load_wicker_vi_bnn, load_wicker_posterior

    model = load_wicker_vi_bnn("path/to/VOGN_MNIST_Posterior", rngs=nnx.Rngs(0))
    posterior = load_wicker_posterior("path/to/VOGN_MNIST_Posterior")

    # or, if you already built the model and just want the posterior:
    posterior = Posterior(model)
"""

from __future__ import annotations

import io
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from models import VI_BNN, VOGNLinear, PriorConfig
from certification import Posterior

Array = jax.Array
VarianceMode = Literal["variance", "std", "raw_s"]


# ---------------------------------------------------------------------------
# 1. Decoding deepbayes' TF-pickled .npy files without TensorFlow
# ---------------------------------------------------------------------------

class _NoTFUnpickler(pickle.Unpickler):
    """Unpickler that turns any tensorflow object back into a numpy array.

    deepbayes stores each weight tensor as
    ``tensorflow.python.framework.ops.convert_to_tensor(<numpy array>)``.
    We only care about the numpy payload, so every tensorflow global is
    replaced by a callable that returns ``np.asarray`` of its first argument.
    """

    def find_class(self, module: str, name: str):
        if module.startswith("tensorflow"):
            return lambda value, *args, **kwargs: np.asarray(value)
        return super().find_class(module, name)


def _read_npy_header(bio: io.BytesIO) -> None:
    """Consume the .npy magic + header from `bio`, leaving the pickle payload.

    Works across numpy versions (the private/public header helpers moved
    around). We only need to advance the stream past the header; the array
    metadata itself is irrelevant because we re-unpickle the payload ourselves.
    """
    import numpy.lib.format as fmt

    major, minor = fmt.read_magic(bio)
    after_magic = bio.tell()

    # Prefer the version-specific public reader; fall back to the private one.
    version_reader = f"read_array_header_{major}_{minor}"
    candidates = [version_reader, "read_array_header_1_0", "_read_array_header"]

    for name in candidates:
        reader = getattr(fmt, name, None)
        if reader is None:
            continue
        bio.seek(after_magic)
        try:
            if name == "_read_array_header":
                reader(bio, (major, minor))
            else:
                reader(bio)
            return
        except Exception:
            continue
    raise RuntimeError("Could not parse .npy header with this numpy version.")


def _load_object_npy_no_tf(path: Path) -> list[np.ndarray]:
    """Load a numpy object-array .npy of TF tensors as a list of np arrays."""
    bio = io.BytesIO(path.read_bytes())
    _read_npy_header(bio)
    obj = _NoTFUnpickler(io.BytesIO(bio.read())).load()
    return [np.asarray(a, dtype=np.float32) for a in obj]


def _try_load_info(path: Path) -> dict:
    """Best-effort load of info.pkl (training config). Returns {} on failure."""
    info_path = path / "info.pkl"
    if not info_path.exists():
        return {}

    class _Lenient(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("tensorflow"):
                return lambda value, *a, **k: np.asarray(value)
            try:
                return super().find_class(module, name)
            except Exception:
                return lambda *a, **k: None

    try:
        with open(info_path, "rb") as f:
            return _Lenient(f).load()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 2. Architecture parsing + weight grouping
# ---------------------------------------------------------------------------

@dataclass
class WickerArch:
    d_in: int
    width: int
    depth: int          # number of hidden layers
    d_out: int
    hidden_widths: list[int] = field(default_factory=list)


def _parse_arch(path: Path) -> WickerArch:
    """Parse arch.json (a Keras Sequential config) into VI_BNN dimensions.

    VI_BNN only supports a single hidden `width` shared by all `depth`
    hidden layers, so we verify every hidden Dense layer has the same width
    and raise a clear error otherwise.
    """
    arch = json.loads((path / "arch.json").read_text())
    layers = arch["config"]["layers"]

    dense = [l for l in layers if l["class_name"] == "Dense"]
    if not dense:
        raise ValueError("arch.json contains no Dense layers.")

    units = [l["config"]["units"] for l in dense]
    *hidden_units, out_units = units

    # d_in comes from the first Dense's declared input, else the InputLayer.
    d_in = None
    for l in layers:
        cfg = l["config"]
        if "batch_input_shape" in cfg and cfg["batch_input_shape"]:
            d_in = int(cfg["batch_input_shape"][-1])
            break
    if d_in is None:
        raise ValueError("Could not determine input dimension from arch.json.")

    if hidden_units and len(set(hidden_units)) != 1:
        raise ValueError(
            "VI_BNN needs one shared hidden width, but arch.json has hidden "
            f"widths {hidden_units}. Load into a custom model instead, or "
            "reshape the network."
        )

    width = hidden_units[0] if hidden_units else out_units
    depth = len(hidden_units)

    return WickerArch(
        d_in=d_in,
        width=width,
        depth=depth,
        d_out=int(out_units),
        hidden_widths=list(hidden_units),
    )


@dataclass
class WickerPosterior:
    """Decoded deepbayes posterior, grouped per Bayesian layer.

    `layers[i]` is a dict with keys mu_w, mu_b, sigma_w, sigma_b (all np
    arrays), ordered [hidden_0, ..., hidden_{depth-1}, output] to match
    VI_BNN / Posterior iteration order.
    """
    arch: WickerArch
    info: dict
    layers: list[dict]


def read_wicker_posterior(
    path: str | Path,
    *,
    variance_mode: VarianceMode = "variance",
    min_sigma: float = 1e-8,
) -> WickerPosterior:
    """Decode a deepbayes posterior directory into grouped mu/sigma arrays.

    This does not touch JAX/flax and has no project dependencies -- useful for
    inspection or for feeding a different model class.
    """
    path = Path(path)
    arch = _parse_arch(path)
    info = _try_load_info(path)

    means = _load_object_npy_no_tf(path / "mean.npy")
    raw_vars = _load_object_npy_no_tf(path / "var.npy")

    if len(means) != len(raw_vars):
        raise ValueError(
            f"mean.npy has {len(means)} tensors but var.npy has {len(raw_vars)}."
        )
    if len(means) % 2 != 0:
        raise ValueError(
            "Expected an even number of tensors (kernel, bias per Dense layer); "
            f"got {len(means)}. Networks without biases are not supported here."
        )

    # Keras get_weights() order: [kernel_0, bias_0, kernel_1, bias_1, ...].
    n_layers = len(means) // 2
    expected = arch.depth + 1
    if n_layers != expected:
        raise ValueError(
            f"Parsed {expected} Bayesian layers from arch.json but found "
            f"{n_layers} weight/bias pairs in mean.npy."
        )

    def to_sigma(v: np.ndarray) -> np.ndarray:
        if variance_mode == "variance":
            sigma = np.sqrt(np.maximum(v, 0.0))
        elif variance_mode == "std":
            sigma = np.abs(v)
        elif variance_mode == "raw_s":
            # bare Fisher accumulator s: std = 1/sqrt(N*s); requires N.
            N = info.get("N")
            if not N:
                raise ValueError(
                    "variance_mode='raw_s' needs 'N' in info.pkl, but it is "
                    "missing. Pass the correct file or use 'variance'."
                )
            sigma = 1.0 / np.sqrt(np.maximum(float(N) * v, 1e-30))
        else:
            raise ValueError(f"Unknown variance_mode: {variance_mode!r}")
        return np.maximum(sigma, min_sigma).astype(np.float32)

    layers: list[dict] = []
    for i in range(n_layers):
        w = np.asarray(means[2 * i], dtype=np.float32)      # (d_in, d_out)
        b = np.asarray(means[2 * i + 1], dtype=np.float32)  # (d_out,)
        vw = np.asarray(raw_vars[2 * i], dtype=np.float32)
        vb = np.asarray(raw_vars[2 * i + 1], dtype=np.float32)

        if w.ndim != 2 or b.ndim != 1 or w.shape[1] != b.shape[0]:
            raise ValueError(
                f"Layer {i} has inconsistent kernel/bias shapes "
                f"{w.shape} / {b.shape}."
            )

        layers.append(
            {
                "mu_w": w,
                "mu_b": b,
                "sigma_w": to_sigma(vw),
                "sigma_b": to_sigma(vb),
            }
        )

    # Shape-check against the parsed architecture (in == out of previous).
    dims = [arch.d_in] + arch.hidden_widths + [arch.d_out]
    for i, layer in enumerate(layers):
        want = (dims[i], dims[i + 1])
        got = layer["mu_w"].shape
        if got != want:
            raise ValueError(
                f"Layer {i} kernel shape {got} does not match architecture "
                f"{want}. Keras kernels are (in, out); no transpose is applied."
            )

    return WickerPosterior(arch=arch, info=info, layers=layers)


# ---------------------------------------------------------------------------
# 3. Loading into a VI_BNN (VOGNLinear)
# ---------------------------------------------------------------------------

def _set_vogn_layer(layer: VOGNLinear, mu_w, mu_b, sigma_w, sigma_b) -> Array:
    """Write mu directly and invert VOGNLinear.sigma() to hit target sigma.

    VOGNLinear.sigma() computes, per element,

        s_hat  = s / s_ema_weight            (when s_ema_weight > 0)
        sigma  = 1 / sqrt(N * s_hat + prec)  with prec = 1/prior.sigma**2

    Fixing s_ema_weight = 1 gives s_hat = s, so to reproduce a target sigma:

        N * s + prec = 1 / sigma**2
        s            = (1/sigma**2 - prec) / N

    which is >= 0 only when sigma <= prior.sigma. We validate that.
    """
    mu_w = jnp.asarray(mu_w, dtype=layer.mu_w.value.dtype)
    mu_b = jnp.asarray(mu_b, dtype=layer.mu_b.value.dtype)
    sigma_w = jnp.asarray(sigma_w)
    sigma_b = jnp.asarray(sigma_b)

    if mu_w.shape != layer.mu_w.value.shape or mu_b.shape != layer.mu_b.value.shape:
        raise ValueError(
            f"Shape mismatch loading a VOGN layer: got mu_w {mu_w.shape}, "
            f"mu_b {mu_b.shape}; layer expects {layer.mu_w.value.shape}, "
            f"{layer.mu_b.value.shape}."
        )

    N = layer.dataset_size
    prec = 1.0 / (layer.prior.sigma ** 2)

    def target_s(sigma: Array) -> Array:
        return (1.0 / (sigma ** 2) - prec) / N

    s_w = target_s(sigma_w)
    s_b = target_s(sigma_b)

    min_s = float(min(jnp.min(s_w), jnp.min(s_b)))
    if min_s < 0.0:
        max_sigma = float(max(jnp.max(sigma_w), jnp.max(sigma_b)))
        raise ValueError(
            "Cannot represent this posterior with the given prior: some target "
            f"sigma ({max_sigma:.4g}) exceeds prior.sigma ({layer.prior.sigma:.4g}), "
            "which would require a negative VOGN curvature. Increase "
            "prior.sigma (e.g. PriorConfig(sigma=...)) or load into a Posterior "
            "with a direct-sigma representation."
        )

    layer.mu_w.value = mu_w
    layer.mu_b.value = mu_b
    layer.s_w.value = s_w
    layer.s_b.value = s_b
    # s_ema_weight = 1 -> s_hat == s (and correction > 0 selects the real path).
    layer.s_ema_weight.value = jnp.asarray(1.0, dtype=layer.s_ema_weight.value.dtype)
    layer.m_ema_weight.value = jnp.asarray(1.0, dtype=layer.m_ema_weight.value.dtype)
    layer.m_w.value = jnp.zeros_like(layer.m_w.value)
    layer.m_b.value = jnp.zeros_like(layer.m_b.value)

    # Return the sigma this layer will now produce, for verification.
    got_w, got_b = layer.sigma()
    err = float(max(
        jnp.max(jnp.abs(got_w - sigma_w)),
        jnp.max(jnp.abs(got_b - sigma_b)),
    ))
    return err


def load_wicker_vi_bnn(
    path: str | Path,
    *,
    rngs: nnx.Rngs,
    prior: PriorConfig | None = None,
    dataset_size: int | None = None,
    variance_mode: VarianceMode = "variance",
    min_sigma: float = 1e-8,
    s_init: float = 1.0,
    verify_tol: float = 1e-4,
) -> VI_BNN:
    """Build a VI_BNN with VOGNLinear layers holding Wicker's posterior.

    Args:
        path: deepbayes posterior directory.
        rngs: nnx.Rngs used only to construct the (immediately overwritten)
            initial parameters.
        prior: Gaussian PriorConfig. Its sigma must be >= every posterior
            sigma. Defaults to PriorConfig(sigma=1.0), wide enough for typical
            trained posteriors.
        dataset_size: N used by VOGNLinear. Defaults to info.pkl's 'N', else
            60000.
        variance_mode: see module docstring; default "variance" (std=sqrt(var)).
        verify_tol: assert the reconstructed sigma matches within this abs tol.

    Returns:
        A VI_BNN whose Posterior(model) reproduces Wicker's (mu, sigma) exactly.
    """
    wp = read_wicker_posterior(path, variance_mode=variance_mode, min_sigma=min_sigma)

    if dataset_size is None:
        dataset_size = int(wp.info.get("N", 60000))
    if prior is None:
        # Wide default so the sigma inversion stays feasible.
        prior = PriorConfig(name="gaussian", sigma=1.0)
    if prior.name != "gaussian":
        raise ValueError("VOGNLinear requires a Gaussian prior.")

    print("Reading pretrained posterior params...")
    model = VI_BNN(
        d_in=wp.arch.d_in,
        d_out=wp.arch.d_out,
        width=wp.arch.width,
        depth=wp.arch.depth,
        rngs=rngs,
        prior=prior,
        layer_cls=VOGNLinear,
        layer_kwargs={"dataset_size": dataset_size, "s_init": s_init},
    )

    target_layers = list(model.layers) + [model.output_layer]
    if len(target_layers) != len(wp.layers):
        raise ValueError(
            f"Model has {len(target_layers)} Bayesian layers but posterior has "
            f"{len(wp.layers)}."
        )

    worst = 0.0
    for layer, src in zip(target_layers, wp.layers):
        err = _set_vogn_layer(
            layer, src["mu_w"], src["mu_b"], src["sigma_w"], src["sigma_b"]
        )
        worst = max(worst, err)

    if worst > verify_tol:
        raise RuntimeError(
            f"Reconstructed VOGN sigma differs from target by {worst:.3g} "
            f"(> tol {verify_tol:g}). This usually means prior.sigma is close "
            "to a posterior sigma; widen the prior."
        )
    
    print("Passed reading checks. Done.")

    return model


def load_wicker_posterior(
    path: str | Path,
    *,
    rngs: nnx.Rngs | None = None,
    prior: PriorConfig | None = None,
    dataset_size: int | None = None,
    variance_mode: VarianceMode = "variance",
    min_sigma: float = 1e-8,
) -> Posterior:
    """Load a deepbayes posterior straight into a `Posterior`.

    Builds the backing VI_BNN (so `posterior.model` is a real, runnable model)
    and returns `Posterior(model)`.
    """
    if rngs is None:
        rngs = nnx.Rngs(0)
    model = load_wicker_vi_bnn(
        path,
        rngs=rngs,
        prior=prior,
        dataset_size=dataset_size,
        variance_mode=variance_mode,
        min_sigma=min_sigma,
    )
    return Posterior(model, min_sigma=min_sigma)


# ---------------------------------------------------------------------------
# 4. CLI: quick inspection of a posterior directory
# ---------------------------------------------------------------------------

def _summary(path: str | Path, variance_mode: VarianceMode = "variance") -> None:
    wp = read_wicker_posterior(path, variance_mode=variance_mode)
    print(f"deepbayes posterior: {path}")
    print(f"  arch : d_in={wp.arch.d_in} width={wp.arch.width} "
          f"depth={wp.arch.depth} d_out={wp.arch.d_out}")
    if wp.info:
        keys = ("N", "classes", "epochs", "learning_rate", "robust_train", "epsilon")
        print("  info :", {k: wp.info.get(k) for k in keys if k in wp.info})
    print(f"  variance_mode = {variance_mode!r}  (std = "
          f"{'sqrt(var)' if variance_mode == 'variance' else variance_mode})")
    for i, layer in enumerate(wp.layers):
        tag = "output" if i == wp.arch.depth else f"hidden_{i}"
        sw = layer["sigma_w"]
        print(f"  layer {i} [{tag:8s}] W{layer['mu_w'].shape} b{layer['mu_b'].shape}  "
              f"sigma_w in [{sw.min():.4g}, {sw.max():.4g}]")

    # Verify a VI_BNN round-trips the posterior exactly.
    model = load_wicker_vi_bnn(path, rngs=nnx.Rngs(0), variance_mode=variance_mode)
    post = Posterior(model)
    max_mu = max(float(jnp.max(jnp.abs(post.mu_w[i] - wp.layers[i]["mu_w"])))
                 for i in range(len(wp.layers)))
    max_sig = max(float(jnp.max(jnp.abs(post.sigma_w[i] - wp.layers[i]["sigma_w"])))
                  for i in range(len(wp.layers)))
    print(f"  round-trip: max|mu err|={max_mu:.2e}  max|sigma err|={max_sig:.2e}  "
          f"(model_type={model.model_type})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect a deepbayes VOGN posterior.")
    ap.add_argument("path", help="posterior directory (with mean.npy/var.npy/arch.json)")
    ap.add_argument("--variance-mode", default="variance",
                    choices=["variance", "std", "raw_s"])
    args = ap.parse_args()
    _summary(args.path, variance_mode=args.variance_mode)