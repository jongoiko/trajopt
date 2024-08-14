from functools import partial
from .util import _unpack, _get_time, _lerp
from .trajectory import Trajectory
from scipy.interpolate import BPoly
import jax
import numpy as np

_ORDER = 2


@partial(jax.jit, static_argnums=(2, 3, 4))
def _collocation_constraints(x_u, time_fractions, dynamics, x_shape, u_shape):
    x, u, t_0, t_f = _unpack(x_u, x_shape, u_shape)
    t = _get_time(t_0, t_f, time_fractions).reshape(-1, 1)
    x_dot = dynamics(x, u, t)
    h = t[1:] - t[:-1]
    constraints = x[1:] - x[:-1] - (h / 2) * (x_dot[1:] + x_dot[:-1])
    return constraints.reshape(-1)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _objective(x_u, time_fractions, running_cost, _, x_shape, u_shape):
    x, u, t_0, t_f = _unpack(x_u, x_shape, u_shape)
    t = _get_time(t_0, t_f, time_fractions).reshape(-1, 1)
    cost = running_cost(x, u, t)
    h = t[1:] - t[:-1]
    return ((h / 2).reshape(-1) * (cost[:-1] + cost[1:])).sum()


class TrapezoidalTrajectory(Trajectory):
    def __init__(self, t, x, u, dynamics):
        super().__init__(t, x, u, dynamics)
        self._x_dot = dynamics(x, u, t)

    def interpolate(self, t):
        x = np.empty((t.size, self._x.shape[1]))
        u = _lerp(t, self._t, self._u)
        x_dot = np.asarray(self._x_dot)
        for i in range(x.shape[1]):
            x[:, i] = BPoly.from_derivatives(
                self._t, np.hstack((self._x[:, [i]], x_dot[:, [i]])), orders=2
            )(t)
        return x, u

    def _approx_dynamics(self, t):
        return _lerp(t, self._t, self._x_dot)
