from functools import partial
from .util import _unpack_x_u, _pack_x_u
from .trajectory import Trajectory
from scipy.interpolate import BPoly
import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _uncompress_x_u(x_u, time_step, dynamics, x_shape, u_shape):
    x, u = _unpack_x_u(x_u, x_shape, u_shape)
    u, u_midpoints = u[::2], u[1::2]
    x_dot = dynamics(x, u)
    x_midpoints = 0.5 * (x[:-1] + x[1:]) + (time_step / 8) * (x_dot[:-1] - x_dot[1:])
    x_dot_midpoints = dynamics(x_midpoints, u_midpoints)
    return x, x_midpoints, u, u_midpoints, x_dot, x_dot_midpoints


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _collocation_constraints(x_u, time_step, dynamics, x_shape, u_shape):
    x, _, _, _, x_dot, x_dot_midpoints = _uncompress_x_u(
        x_u, time_step, dynamics, x_shape, u_shape
    )
    constraints = (
        x[1:]
        - x[:-1]
        - (time_step / 6) * (x_dot[1:] + 4 * x_dot_midpoints + x_dot[:-1])
    )
    return constraints.reshape(-1)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def _objective(x_u, time_step, running_cost, dynamics, x_shape, u_shape):
    x, x_midpoints, u, u_midpoints, _, _ = _uncompress_x_u(
        x_u, time_step, dynamics, x_shape, u_shape
    )
    cost, cost_midpoints = running_cost(x, u), running_cost(x_midpoints, u_midpoints)
    return (time_step / 6) * (cost[:-1] + 4 * cost_midpoints + cost[1:]).sum()


class HermiteSimpsonTrajectory(Trajectory):
    def __init__(self, t, x, u, dynamics):
        super().__init__(t, x, u, dynamics)
        time_step = float(t[1] - t[0])
        (
            self._x,
            self._x_midpoints,
            self._u,
            self._u_midpoints,
            self._x_dot,
            self._x_dot_midpoints,
        ) = _uncompress_x_u(
            _pack_x_u(x, u), time_step, dynamics, tuple(x.shape), tuple(u.shape)
        )

    def interpolate(self, t):
        x = np.empty((t.size, self._x.shape[1]))
        u = np.empty((t.size, self._u.shape[1]))
        for i, t_val in enumerate(t):
            knot_index = jnp.argmax(t_val < self._t) - 1
            p = jnp.polyfit(
                jnp.linspace(self._t[knot_index], self._t[knot_index + 1], 3),
                jnp.vstack(
                    [
                        self._u[knot_index],
                        self._u_midpoints[knot_index],
                        self._u[knot_index + 1],
                    ]
                ),
                deg=2,
            )
            u[i, :] = jnp.polyval(p, t_val)
        t_full = np.linspace(self._t[0], self._t[-1], self._t.size * 2 - 1)
        x_full = np.repeat(np.asarray(self._x), 2, axis=0)[:-1]
        x_full[1::2] = self._x_midpoints
        x_dot_full = np.repeat(np.asarray(self._x_dot), 2, axis=0)[:-1]
        x_dot_full[1::2] = self._x_dot_midpoints
        for i in range(x.shape[1]):
            x[:, i] = BPoly.from_derivatives(
                t_full, np.hstack((x_full[:, [i]], x_dot_full[:, [i]])), orders=3
            )(t)
        return x, u
