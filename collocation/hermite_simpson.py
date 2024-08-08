from functools import partial
from .util import _unpack, _pack, _get_time
from .trajectory import Trajectory
from scipy.interpolate import BPoly
import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(2, 3, 4))
def _uncompress_x_u(x_u, time_fractions, dynamics, x_shape, u_shape):
    x, u, t_0, t_f = _unpack(x_u, x_shape, u_shape)
    u, u_midpoints = u[::2], u[1::2]
    t = _get_time(t_0, t_f, time_fractions).reshape(-1, 1)
    h = t[1:] - t[:-1]
    x_dot = dynamics(x, u, t)
    x_midpoints = 0.5 * (x[:-1] + x[1:]) + (h / 8) * (x_dot[:-1] - x_dot[1:])
    x_dot_midpoints = dynamics(x_midpoints, u_midpoints, t[:-1] + h / 2)
    return x, x_midpoints, u, u_midpoints, x_dot, x_dot_midpoints, t, h


@partial(jax.jit, static_argnums=(2, 3, 4))
def _collocation_constraints(x_u, time_fractions, dynamics, x_shape, u_shape):
    x, _, _, _, x_dot, x_dot_midpoints, _, h = _uncompress_x_u(
        x_u, time_fractions, dynamics, x_shape, u_shape
    )
    constraints = (
        x[1:] - x[:-1] - (h / 6) * (x_dot[1:] + 4 * x_dot_midpoints + x_dot[:-1])
    )
    return constraints.reshape(-1)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _objective(x_u, time_fractions, running_cost, dynamics, x_shape, u_shape):
    x, x_midpoints, u, u_midpoints, _, _, t, h = _uncompress_x_u(
        x_u, time_fractions, dynamics, x_shape, u_shape
    )

    cost, cost_midpoints = running_cost(x, u, t), running_cost(
        x_midpoints, u_midpoints, t[:-1] + h / 2
    )
    return ((h / 6).reshape(-1) * (cost[:-1] + 4 * cost_midpoints + cost[1:])).sum()


def _interp_quadratic_midpoints(values, values_midpoints, t_val, t):
    knot_index = jnp.argmax(t_val < t) - 1
    p = jnp.polyfit(
        jnp.linspace(t[knot_index], t[knot_index + 1], 3),
        jnp.vstack(
            [
                values[knot_index],
                values_midpoints[knot_index],
                values[knot_index + 1],
            ]
        ),
        deg=2,
    )
    return jnp.polyval(p, t_val)


class HermiteSimpsonTrajectory(Trajectory):
    def __init__(self, t, x, u, dynamics):
        super().__init__(t, x, u, dynamics)
        self._interp_quadratic_midpoints = jax.jit(
            jax.vmap(_interp_quadratic_midpoints, in_axes=(None, None, 0, None))
        )
        time_fractions = (t - self.t_0) / (self.t_f - self.t_0)
        (
            self._x,
            self._x_midpoints,
            self._u,
            self._u_midpoints,
            self._x_dot,
            self._x_dot_midpoints,
            _,
            self._h,
        ) = _uncompress_x_u(
            _pack(x, u, self.t_0, self.t_f),
            time_fractions,
            dynamics,
            tuple(x.shape),
            tuple(u.shape),
        )

    def interpolate(self, t):
        x = np.empty((t.size, self._x.shape[1]))
        u = self._interp_quadratic_midpoints(self._u, self._u_midpoints, t, self._t)
        t_full = np.repeat(np.asarray(self._t), 2)[:-1]
        t_full[1::2] = self._t[:-1] + self._h.reshape(-1) / 2
        x_full = np.repeat(np.asarray(self._x), 2, axis=0)[:-1]
        x_full[1::2] = self._x_midpoints
        x_dot_full = np.repeat(np.asarray(self._x_dot), 2, axis=0)[:-1]
        x_dot_full[1::2] = self._x_dot_midpoints
        for i in range(x.shape[1]):
            x[:, i] = BPoly.from_derivatives(
                t_full, np.hstack((x_full[:, [i]], x_dot_full[:, [i]])), orders=3
            )(t)
        return x, u
