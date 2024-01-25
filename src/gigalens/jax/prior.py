from gigalens import prior
from gigalens.jax.physical_model import PhysicalModel
from tensorflow_probability.substrates.jax import distributions as tfd, bijectors as tfb
from typing import List, Optional
from jax import random


class ProfilePrior(prior.ProfilePriorBase):

    _tfd = tfd

    def __init__(self, profile, params):
        super(ProfilePrior, self).__init__(profile, params)


class CompoundPrior(prior.CompoundPriorBase):

    _tfd = tfd

    def __init__(self, models: Optional[List[ProfilePrior]]):
        super(CompoundPrior, self).__init__(models)


class LensPrior(prior.LensPriorBase):

    _compound_prior_cls = CompoundPrior
    _phys_model_cls = PhysicalModel
    _tfd = tfd
    _tfb = tfb

    def __init__(self,
                 lenses: Optional[List[ProfilePrior]] = None,
                 sources: Optional[List[ProfilePrior]] = None,
                 foreground: Optional[List[ProfilePrior]] = None):

        if foreground is None:
            foreground = []
        if sources is None:
            sources = []
        if lenses is None:
            lenses = []

        super(LensPrior, self).__init__(lenses, sources, foreground)

    def make_seed(self, seed):
        return random.PRNGKey(seed)
