from gigalens.prior import ProfilePrior, CompoundPrior, LensPrior, Prior, make_prior_and_model
from gigalens.jax.physical_model import PhysicalModel
from tensorflow_probability.substrates.jax import distributions as tfd, bijectors as tfb
from jax import random


ProfilePrior._tfd = tfd
CompoundPrior._tfd = tfd
LensPrior._tfd = tfd
LensPrior._tfb = tfb
LensPrior._seed = random.PRNGKey(0)
LensPrior._phys_model_cls = PhysicalModel

__all__ = ["Prior", "make_prior_and_model"]
