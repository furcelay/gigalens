from gigalens import prior
from gigalens.jax.model import PhysicalModel
from tensorflow_probability.substrates.jax import distributions as tfd, bijectors as tfb


class ProfilePrior(prior.ProfilePrior):

    _tfd = tfd

    def __int__(self, profile, params):
        super(ProfilePrior, self).__init__(profile, params)


class CompoundPrior(prior.CompoundPrior):

    _tfd = tfd

    def __int__(self, models: ProfilePrior = None):
        super(CompoundPrior, self).__init__(models)


class LensPrior(prior.LensPrior):

    _phys_model_cls = PhysicalModel
    _tfd = tfd
    _tfb = tfb

    def __int__(self, lenses=CompoundPrior(), sources=CompoundPrior(), foreground=CompoundPrior()):
        super(LensPrior, self).__init__(lenses, sources, foreground)
