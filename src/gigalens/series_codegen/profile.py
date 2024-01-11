from abc import ABC, abstractmethod
from gigalens.profile import Parameterized
from sympy import symbols, Matrix


class SPMassProfile(Parameterized, ABC):
    """Interface for a mass profile symbolic expression."""

    _series_var = ''

    def __init__(self, *args, **kwargs):
        super(SPMassProfile, self).__init__(*args, **kwargs)
        self.args = symbols(' '.join(self.params))
        self.series_var = self.args[self.params.index(self._series_var)]

    @abstractmethod
    def deriv(self, x, y, *args):
        pass

    def hessian(self, x, y, *args):
        f_x, f_y = self.deriv(x, y, *args)
        f_xx = f_x.diff(x)
        f_xy = f_x.diff(y)
        f_yy = f_y.diff(y)
        return Matrix([f_xx, f_xy, f_xy, f_yy])
