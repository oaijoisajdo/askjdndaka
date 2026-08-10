from .gaussian import GaussianPosterior
from .mcd_posteriors import DropoutPosterior


class Posterior:
    """Factory dispatching to the posterior matching the model type."""
    def __new__(cls, model, **kwargs):
        if getattr(model, "model_type", None) in ("mc_dropout", "deterministic"):
            return DropoutPosterior(model, **kwargs)
        return GaussianPosterior(model, **kwargs)