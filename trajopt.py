import cyipopt
import numpy as np
import jax
import jax.numpy as jnp
from collocation.util import _pack_x_u, _unpack_x_u
from collocation import trapezoidal


class Guess:
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
    METHOD_ALIASES = {
        "trapezoidal": (trapezoidal, trapezoidal.TrapezoidalTrajectory),
    }

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
        method,
        solver_kwargs=None,
    ):
        if method not in self.METHOD_ALIASES:
            raise ValueError(
                f"Choose one collocation method from {list(self.METHOD_ALIASES.keys())}"
            )
        self._method = self.METHOD_ALIASES[method][0]
        self._trajectory_subclass = self.METHOD_ALIASES[method][1]
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
        self.objective = lambda x_u: self._method._objective(
            x_u, self._time_step, self._running_cost, self._x_shape, self._u_shape
        )
        self.gradient = jax.jit(jax.grad(self.objective))
        self.constraints = lambda x_u: self._method._collocation_constraints(
            x_u, self._time_step, self._dynamics, self._x_shape, self._u_shape
        )
        self._n_grid = n_grid
        self._t = jnp.linspace(self._t_0, self._t_f, self._n_grid)
        self._initial_guess = _pack_x_u(*initial_guess.interpolate(self._t))
        self._jacobian_structure = self._estimate_jacobian_structure()
        self.jacobianstructure = jax.jit(lambda: self._jacobian_structure)
        self.jacobian = jax.jit(
            lambda x_u: jax.jacobian(self.constraints)(x_u)[self._jacobian_structure]
        )
        self._nlp = self._build_nlp()
        if solver_kwargs is None:
            return
        for key, value in solver_kwargs.items():
            self._nlp.add_option(key, value)

    def _estimate_jacobian_structure(self, seed=42, n_samples=100):
        key = jax.random.key(seed)
        key, *subkeys = jax.random.split(key, n_samples)
        get_jacobian = jax.jit(jax.jacobian(self.constraints))
        jac_sum = np.zeros(get_jacobian(self._initial_guess).shape)
        for subkey in subkeys:
            perturbation = jax.random.normal(subkey, shape=self._initial_guess.shape)
            jacobian = get_jacobian(self._initial_guess + perturbation)
            jac_sum += jnp.abs(jacobian)
        return jnp.nonzero(jac_sum)

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
        return self._trajectory_subclass(self._t, x, u, self._dynamics(x, u))
