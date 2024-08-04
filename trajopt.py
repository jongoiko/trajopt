from functools import partial
import cyipopt
import numpy as np
import jax
import jax.numpy as jnp


@jax.jit
def _pack_x_u(x, u):
    return jnp.concatenate([x.reshape(-1), u.reshape(-1)])


@partial(jax.jit, static_argnums=(1, 2))
def _unpack_x_u(x_u, x_shape, u_shape):
    num_x_vars = np.prod(np.asarray(x_shape))
    return x_u[:num_x_vars].reshape(*x_shape), x_u[num_x_vars:].reshape(*u_shape)


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


class Trajectory:
    def __init__(self, t, x, u):
        self._t = t
        self._x = x
        self._u = u

    def interpolate(self, t):
        x = np.empty((t.size, self._x.shape[1]))
        u = np.empty((t.size, self._u.shape[1]))
        for i in range(x.shape[1]):
            x[:, i] = np.interp(t, self._t, self._x[:, i])
        for i in range(u.shape[1]):
            u[:, i] = np.interp(t, self._t, self._u[:, i])
        return x, u


class OCP:
    def __init__(
        self,
        dynamics,
        running_cost,
        t_0,
        t_f,
        x_0,
        x_f,
        x_lower,
        x_upper,
        u_lower,
        u_upper,
        initial_guess,
        n_grid,
        solver_kwargs=None,
    ):
        self._dynamics = jax.jit(jax.vmap(dynamics, in_axes=(0, 0)))
        self._running_cost = jax.jit(jax.vmap(running_cost, in_axes=(0, 0)))
        self._t_0 = t_0
        self._t_f = t_f
        self._x_0 = x_0
        self._x_f = x_f
        self._x_lower = x_lower
        self._x_upper = x_upper
        self._u_lower = u_lower
        self._u_upper = u_upper
        self._x_shape = (n_grid, x_0.size)
        self._u_shape = (n_grid, initial_guess._u.shape[1])
        self._time_step = (t_f - t_0) / (n_grid - 1)
        self.constraints = lambda x_u: _collocation_constraints(
            x_u, self._time_step, self._dynamics, self._x_shape, self._u_shape
        )
        self.objective = lambda x_u: _objective(
            x_u, self._time_step, self._running_cost, self._x_shape, self._u_shape
        )
        self.gradient = jax.jit(jax.grad(self.objective))
        self.jacobian = jax.jit(jax.jacobian(self.constraints))
        self._n_grid = n_grid
        self._t = jnp.linspace(self._t_0, self._t_f, self._n_grid)
        self._initial_guess = _pack_x_u(*initial_guess.interpolate(self._t))
        self._nlp = self._build_nlp()
        if solver_kwargs is None:
            return
        for key, value in solver_kwargs.items():
            self._nlp.add_option(key, value)

    def _build_nlp(self):
        n = self._initial_guess.size
        m = self.constraints(self._initial_guess).size
        lb_x, ub_x, lb_u, ub_u = [
            jnp.tile(arr, (self._n_grid, 1))
            for arr in [self._x_lower, self._x_upper, self._u_lower, self._u_upper]
        ]
        lb_x = lb_x.at[0].set(self._x_0).at[-1].set(self._x_f)
        ub_x = ub_x.at[0].set(self._x_0).at[-1].set(self._x_f)
        lb, ub = _pack_x_u(lb_x, lb_u), _pack_x_u(ub_x, ub_u)
        zeros = jnp.zeros(m)
        return cyipopt.Problem(
            problem_obj=self, n=n, m=m, cl=zeros, cu=zeros, lb=lb, ub=ub
        )

    def solve(self):
        x_u, _ = self._nlp.solve(self._initial_guess)
        x, u = _unpack_x_u(x_u, self._x_shape, self._u_shape)
        self._nlp.close()
        return Trajectory(self._t, x, u)
