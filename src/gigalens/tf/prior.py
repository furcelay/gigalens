from gigalens import prior
from gigalens.tf.model import PhysicalModel
from tensorflow_probability import distributions as tfd, bijectors as tfb
from typing import Optional


class ProfilePrior(prior.ProfilePriorBase):

    _tfd = tfd

    def __init__(self, profile, params):
        super(ProfilePrior, self).__init__(profile, params)


class CompoundPrior(prior.CompoundPriorBase):

    _tfd = tfd

    def __init__(self, models: Optional[ProfilePrior] = None):
        super(CompoundPrior, self).__init__(models)


class LensPrior(prior.LensPriorBase):

    _phys_model_cls = PhysicalModel
    _tfd = tfd
    _tfb = tfb

    def __init__(self, lenses=CompoundPrior(), sources=CompoundPrior(), foreground=CompoundPrior()):
        super(LensPrior, self).__init__(lenses, sources, foreground)
