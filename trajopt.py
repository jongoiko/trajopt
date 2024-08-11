import cyipopt
import numpy as np
import jax
import jax.numpy as jnp
import scipy.integrate
import types
from functools import partial
from collocation.util import _pack, _unpack, _get_time
from collocation import trapezoidal, hermite_simpson


class Guess:
    def __init__(self, t, x, u):
        self._t = t
        self._x = x
        self._u = u

    def interpolate(self, t, u_midpoints=False):
        x = np.empty((t.size, self._x.shape[1]))
        for i in range(x.shape[1]):
            x[:, i] = np.interp(t, self._t, self._x[:, i])
        u_t = (
            t
            if not u_midpoints
            else np.interp(
                np.linspace(0, t.size - 1, 2 * t.size - 1), np.arange(t.size), t
            )
        )
        u = np.empty((u_t.size, self._u.shape[1]))
        for i in range(u.shape[1]):
            u[:, i] = np.interp(u_t, self._t, self._u[:, i])
        return x, u

    @staticmethod
    def from_trajectory(trajectory):
        return Guess(trajectory._t, trajectory._x, trajectory._u)


@partial(jax.jit, static_argnums=range(2, 8))
def _objective(
    x_u,
    time_fractions,
    running_cost_integrator,
    running_cost_func,
    terminal_cost_func,
    dynamics,
    x_shape,
    u_shape,
):
    running_cost = running_cost_integrator(
        x_u, time_fractions, running_cost_func, dynamics, x_shape, u_shape
    )
    x, _, t_0, t_f = _unpack(x_u, x_shape, u_shape)
    terminal_cost = terminal_cost_func(t_0, x[0], t_f, x[-1])
    return running_cost + terminal_cost


class OCP:
    _METHOD_ALIASES = {
        "trapezoidal": (trapezoidal, trapezoidal.TrapezoidalTrajectory, False),
        "hermite-simpson": (
            hermite_simpson,
            hermite_simpson.HermiteSimpsonTrajectory,
            True,
        ),
    }

    def __init__(
        self,
        dynamics,
        running_cost,
        t_0,
        t_f,
        x_0,
        x_f,
        x_bounds,
        u_bounds,
        initial_guess,
        n_grid,
        method,
        terminal_cost=None,
        solver_kwargs=None,
        jac_sparsity_estimation_samples=100,
    ):
        if method not in self._METHOD_ALIASES:
            raise ValueError(
                f"Choose one collocation method from {list(self._METHOD_ALIASES.keys())}"
            )
        (
            self._method,
            self._trajectory_subclass,
            self._u_midpoints,
        ) = self._METHOD_ALIASES[method]
        self._dynamics = jax.jit(jax.vmap(dynamics, in_axes=(0, 0, 0)))
        self._running_cost = jax.vmap(running_cost, in_axes=(0, 0, 0))
        self._terminal_cost = jax.jit(
            (lambda t_0, x_0, t_f, x_f: 0) if terminal_cost is None else terminal_cost
        )
        self._t_0_lower, self._t_0_upper = t_0
        self._t_f_lower, self._t_f_upper = t_f
        self._x_0_lower, self._x_0_upper = x_0
        self._x_f_lower, self._x_f_upper = x_f
        self._x_lower, self._x_upper = x_bounds
        self._u_lower, self._u_upper = u_bounds
        self._time_fractions = jnp.linspace(0, 1, n_grid)
        self._initial_guess = initial_guess
        self._solver_kwargs = solver_kwargs
        self._jac_sparsity_estimation_samples = jac_sparsity_estimation_samples

    def _estimate_jacobian_structure(
        self, n_samples, initial_x_u, constraints, seed=42
    ):
        key = jax.random.key(seed)
        key, *subkeys = jax.random.split(key, n_samples)
        get_jacobian = jax.jit(jax.jacobian(constraints))
        jac_sum = np.zeros(get_jacobian(initial_x_u).shape)
        for subkey in subkeys:
            perturbation = jax.random.normal(subkey, shape=initial_x_u.shape)
            jacobian = get_jacobian(initial_x_u + perturbation)
            jac_sum += jnp.abs(jacobian)
        return jnp.nonzero(jac_sum)

    def _pack_initial_guess(self):
        guess_t_0, guess_t_f = (
            self._initial_guess._t.min(),
            self._initial_guess._t.max(),
        )
        return _pack(
            *self._initial_guess.interpolate(
                _get_time(guess_t_0, guess_t_f, self._time_fractions),
                self._u_midpoints,
            ),
            guess_t_0,
            guess_t_f,
        )

    def _get_x_u_shapes(self):
        n_grid = self._time_fractions.size
        x_shape = (n_grid, self._x_0_lower.size)
        u_shape = (
            n_grid if not self._u_midpoints else 2 * n_grid - 1,
            self._u_lower.size,
        )
        return x_shape, u_shape

    def _build_nlp(self, packed_initial_guess):
        n_grid = self._time_fractions.size
        x_shape, u_shape = self._get_x_u_shapes()
        problem = types.SimpleNamespace()
        problem.objective = lambda x_u: _objective(
            x_u,
            self._time_fractions,
            self._method._objective,
            self._running_cost,
            self._terminal_cost,
            self._dynamics,
            x_shape,
            u_shape,
        )
        problem.gradient = jax.jit(jax.grad(problem.objective))
        constraints = lambda x_u: self._method._collocation_constraints(
            x_u, self._time_fractions, self._dynamics, x_shape, u_shape
        )
        problem.constraints = constraints
        jacobian_structure = self._estimate_jacobian_structure(
            self._jac_sparsity_estimation_samples, packed_initial_guess, constraints
        )
        problem.jacobianstructure = jax.jit(lambda: jacobian_structure)
        problem.jacobian = jax.jit(
            lambda x_u: jax.jacobian(constraints)(x_u)[jacobian_structure]
        )
        n = packed_initial_guess.size
        m = problem.constraints(packed_initial_guess).size
        lb_x, ub_x = [
            jnp.tile(arr, (n_grid, 1)) for arr in [self._x_lower, self._x_upper]
        ]
        lb_u, ub_u = [
            jnp.tile(
                arr,
                (n_grid if not self._u_midpoints else 2 * n_grid - 1, 1),
            )
            for arr in [self._u_lower, self._u_upper]
        ]
        lb_x = lb_x.at[0].set(self._x_0_lower).at[-1].set(self._x_f_lower)
        ub_x = ub_x.at[0].set(self._x_0_upper).at[-1].set(self._x_f_upper)
        lb, ub = _pack(lb_x, lb_u, self._t_0_lower, self._t_f_lower), _pack(
            ub_x, ub_u, self._t_0_upper, self._t_f_upper
        )
        zeros = jnp.zeros(m)
        problem = cyipopt.Problem(
            problem_obj=problem, n=n, m=m, cl=zeros, cu=zeros, lb=lb, ub=ub
        )
        if self._solver_kwargs is None:
            return problem
        for key, value in self._solver_kwargs.items():
            problem.add_option(key, value)
        return problem

    def solve(self):
        initial_x_u = self._pack_initial_guess()
        nlp = self._build_nlp(initial_x_u)
        x_u, _ = nlp.solve(initial_x_u)
        nlp.close()
        x_shape, u_shape = self._get_x_u_shapes()
        x, u, t_0, t_f = _unpack(x_u, x_shape, u_shape)
        t = _get_time(t_0, t_f, self._time_fractions)
        return self._trajectory_subclass(t, x, u, self._dynamics)
