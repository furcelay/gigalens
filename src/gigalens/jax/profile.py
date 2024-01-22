import functools
from jax import jit, vjp, vmap
from jax.tree_util import Partial
from jax import numpy as jnp
import gigalens.profile
from abc import ABC


class MassProfile(gigalens.profile.MassProfile, ABC):
    """Tensorflow interface for a mass profile."""

    @functools.partial(jit, static_argnums=(0,))
    def hessian(self, x, y, **kwargs):
        """Calculates hessian with autograd in reverse mode.

                Args:
                    x: :math:`x` coordinate at which to evaluate the deflection
                    y: :math:`y` coordinate at which to evaluate the deflection
                    **kwargs: Mass profile parameters. Each parameter must be shaped in a way that is broadcastable with x and y

                Returns:
                    A tuple :math:`(\\f_xx, \\f_xy, \\f_yx, \\f_yy)` containing the hessian matrix in the :math:`x` and :math:`y` directions
        """

        partial_deriv = Partial(self.deriv, **kwargs)
        _, vjp_deriv = vjp(partial_deriv, x, y)
        std_basis = (
            jnp.stack([jnp.ones_like(x), jnp.zeros_like(x)]),
            jnp.stack([jnp.zeros_like(x), jnp.ones_like(x)])
        )
        (f_xx, f_yx), (f_xy, f_yy) = vmap(vjp_deriv, in_axes=0, out_axes=0)(std_basis)
        return f_xx, f_xy, f_yx, f_yy

    @functools.partial(jit, static_argnums=(0,))
    def convergence(self, x, y, **kwargs):
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, **kwargs)
        kappa = (f_xx + f_yy) / 2
        return kappa

    @functools.partial(jit, static_argnums=(0,))
    def shear(self, x, y, **kwargs):
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, **kwargs)
        gamma1 = (f_xx - f_yy) / 2
        gamma2 = f_xy
        return gamma1, gamma2
