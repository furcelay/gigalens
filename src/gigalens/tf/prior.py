from gigalens.prior import ProfilePrior, CompoundPrior, LensPrior, Prior, make_prior_and_model
from gigalens.tf.physical_model import PhysicalModel
from tensorflow_probability import distributions as tfd, bijectors as tfb

ProfilePrior._tfd = tfd
CompoundPrior._tfd = tfd
LensPrior._tfd = tfd
LensPrior._tfb = tfb
LensPrior._seed = 0
LensPrior._phys_model_cls = PhysicalModel

__all__ = ["Prior", "make_prior_and_model"]
