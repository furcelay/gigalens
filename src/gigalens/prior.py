from typing import Optional, List


class ProfilePriorBase:
    _tfd = None

    def __init__(self, profile, params):
        self.profile = profile
        self.variables = {}
        self.constants = {}
        self.prior = None
        self.num_free_params = 0
        for k in params.keys():
            if k not in profile.params:
                raise RuntimeError(f"Unknown parameter '{k}' for model {profile}.")
        for k in profile.params:
            try:
                p = params[k]
            except KeyError:
                raise RuntimeError(f"Missing parameter '{k}' for model {profile}.")
            if isinstance(p, self._tfd.Distribution):
                self.variables[k] = p
                self.num_free_params += 1
            else:
                try:
                    self.constants[k] = float(p)
                except ValueError:
                    raise RuntimeError(f"Invalid value {p} for parameter '{k}', should be number or tfp distribution.")
        if self.variables:
            self.prior = self._tfd.JointDistributionNamed(self.variables)

    def __repr__(self):
        return f"{self.profile}(vars:{list(self.variables.keys())},const:{list(self.constants.keys())})"


class CompoundPriorBase:
    """
        lenses:    {1: prof1,         2: prof2,    ...}
        prior:     {1: {p1, p2 , p3}, 2: {p1, p2}, ...}
        constants: {1: {p4},          2: {},       ...}
    """

    _tfd = None

    def __init__(self, *models: Optional[ProfilePriorBase]):
        self.models = models
        self.keys = [str(i) for i in range(len(models))]
        self.profiles = {str(i): m.profile for i, m in enumerate(models)}
        self.constants = {str(i): m.constants for i, m in enumerate(models)}
        self.num_free_params = 0
        for m in models:
            self.num_free_params += m.num_free_params

        priors = {str(i): m.prior for i, m in enumerate(models) if m.prior is not None}
        self.prior = None
        if priors:
            self.prior = self._tfd.JointDistributionNamed(priors)

    def __repr__(self):
        return f"CompoundModel({self.models})"


class LensPriorBase:
    _phys_model_cls = None
    _tfd = None
    _tfb = None

    def __init__(self, lenses: CompoundPriorBase, sources: CompoundPriorBase, foreground: CompoundPriorBase):
        self.lenses = lenses
        self.sources = sources
        self.foreground = foreground
        self.num_free_params = lenses.num_free_params + sources.num_free_params + foreground.num_free_params

        priors = {}
        if lenses.prior is not None:
            priors['lenses'] = lenses.prior
        if sources.prior is not None:
            priors['sources'] = sources.prior
        if foreground.prior is not None:
            priors['foreground'] = foreground.prior

        self.prior = None
        if priors:
            prior = self._tfd.JointDistributionNamed(priors)
            example = prior.sample()
            size = self.num_free_params
            self.pack_bij = self._tfb.Chain([
                self._tfb.pack_sequence_as(example),
                self._tfb.Split(size),
                self._tfb.Reshape(event_shape_out=(-1,), event_shape_in=(size, -1)),
                self._tfb.Transpose(perm=(1, 0)),
            ])
            self.unconstraining_bij = prior.experimental_default_event_space_bijector()
            self.bij = self._tfb.Chain([self.unconstraining_bij, self.pack_bij])

    def get_physical_model(self):
        return self._phys_model_cls(
            self.lenses.profiles,
            self.sources.profiles,
            self.foreground.profiles,
            self.lenses.constants,
            self.sources.constants,
            self.foreground.constants
        )

    def sample(self, shape=(1,), seed=None):
        return self.prior.sample(shape, seed)

    def __repr__(self):
        return f"lenses: {self.lenses} | sources: {self.sources} | foreground: {self.foreground}"
