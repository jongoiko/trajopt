from functools import partial
from .util import _unpack_x_u
from .trajectory import Trajectory
from scipy.interpolate import BPoly
import jax
import numpy as np


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _collocation_constraints(x_u, time_step, dynamics, x_shape, u_shape):
    x, u = _unpack_x_u(x_u, x_shape, u_shape)
    x_dot = dynamics(x, u)
    constraints = x[1:] - x[:-1] - (time_step / 2) * (x_dot[1:] + x_dot[:-1])
    return constraints.reshape(-1)


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _objective(x_u, time_step, running_cost, x_shape, u_shape):
    x, u = _unpack_x_u(x_u, x_shape, u_shape)
    cost = running_cost(x, u)
    return (time_step / 2) * (cost[:-1] + cost[1:]).sum()


class TrapezoidalTrajectory(Trajectory):
    def __init__(self, t, x, u, x_dot):
        super().__init__(t, x, u, x_dot)

    def interpolate(self, t):
        x = np.empty((t.size, self._x.shape[1]))
        u = np.empty((t.size, self._u.shape[1]))
        for i in range(u.shape[1]):
            u[:, i] = np.interp(t, self._t, self._u[:, i])
        x_dot = np.asarray(self._x_dot)
        for i in range(x.shape[1]):
            x[:, i] = BPoly.from_derivatives(
                self._t, np.hstack((self._x[:, [i]], x_dot[:, [i]])), orders=2
            )(t)
        return x, u
