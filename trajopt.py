import cyipopt
import numpy as np
import jax
import jax.numpy as jnp
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
        self._t_0_lower, self._t_0_upper = t_0
        self._t_f_lower, self._t_f_upper = t_f
        self._x_0_lower, self._x_0_upper = x_0
        self._x_f_lower, self._x_f_upper = x_f
        self._x_lower, self._x_upper = x_bounds
        self._u_lower, self._u_upper = u_bounds
        self._x_shape = (n_grid, self._x_0_lower.size)
        self._u_shape = (
            n_grid if not self._u_midpoints else 2 * n_grid - 1,
            initial_guess._u.shape[1],
        )
        self._time_fractions = jnp.linspace(0, 1, n_grid)
        self.objective = lambda x_u: self._method._objective(
            x_u,
            self._time_fractions,
            self._running_cost,
            self._dynamics,
            self._x_shape,
            self._u_shape,
        )
        self.gradient = jax.jit(jax.grad(self.objective))
        self.constraints = lambda x_u: self._method._collocation_constraints(
            x_u, self._time_fractions, self._dynamics, self._x_shape, self._u_shape
        )
        self._n_grid = n_grid
        guess_t_0, guess_t_f = initial_guess._t.min(), initial_guess._t.max()
        self._initial_guess = _pack(
            *initial_guess.interpolate(
                jnp.linspace(guess_t_0, guess_t_f, n_grid),
                self._u_midpoints,
            ),
            guess_t_0,
            guess_t_f,
        )
        self._jacobian_structure = self._estimate_jacobian_structure(
            jac_sparsity_estimation_samples
        )
        self.jacobianstructure = jax.jit(lambda: self._jacobian_structure)
        self.jacobian = jax.jit(
            lambda x_u: jax.jacobian(self.constraints)(x_u)[self._jacobian_structure]
        )
        self._nlp = self._build_nlp()
        if solver_kwargs is None:
            return
        for key, value in solver_kwargs.items():
            self._nlp.add_option(key, value)

    def _estimate_jacobian_structure(self, n_samples, seed=42):
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
        lb_x, ub_x = [
            jnp.tile(arr, (self._n_grid, 1)) for arr in [self._x_lower, self._x_upper]
        ]
        lb_u, ub_u = [
            jnp.tile(
                arr,
                (self._n_grid if not self._u_midpoints else 2 * self._n_grid - 1, 1),
            )
            for arr in [self._u_lower, self._u_upper]
        ]
        lb_x = lb_x.at[0].set(self._x_0_lower).at[-1].set(self._x_f_lower)
        ub_x = ub_x.at[0].set(self._x_0_upper).at[-1].set(self._x_f_upper)
        lb, ub = _pack(lb_x, lb_u, self._t_0_lower, self._t_f_lower), _pack(
            ub_x, ub_u, self._t_0_upper, self._t_f_upper
        )
        zeros = jnp.zeros(m)
        return cyipopt.Problem(
            problem_obj=self, n=n, m=m, cl=zeros, cu=zeros, lb=lb, ub=ub
        )

    def solve(self):
        x_u, _ = self._nlp.solve(self._initial_guess)
        x, u, t_0, t_f = _unpack(x_u, self._x_shape, self._u_shape)
        self._nlp.close()
        t = _get_time(t_0, t_f, self._time_fractions)
        return self._trajectory_subclass(t, x, u, self._dynamics)
