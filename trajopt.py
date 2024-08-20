import cyipopt
import numpy as np
import jax
import jax.numpy as jnp
import scipy.integrate
import types
import logging
from typing import Callable, Optional, Tuple, Any
from functools import partial
from collocation.util import _pack, _unpack, _get_time, _lerp
from collocation.trajectory import Trajectory
from collocation import trapezoidal, hermite_simpson


logger = logging.getLogger(__name__)


class Guess:
    def __init__(self, t: jax.Array, x: jax.Array, u: jax.Array):
        self._t = t
        self._x = x
        self._u = u

    def interpolate(self, t: jax.Array, u_midpoints: bool = False):
        x = _lerp(t, self._t, self._x)
        u_t = (
            t
            if not u_midpoints
            else np.interp(
                np.linspace(0, t.size - 1, 2 * t.size - 1), np.arange(t.size), t
            )
        )
        u = _lerp(u_t, self._t, self._u)
        return x, u

    @staticmethod
    def from_trajectory(trajectory: Trajectory):
        return Guess(trajectory._t, trajectory._x, trajectory._u)


@partial(jax.jit, static_argnums=range(2, 8))
def _objective(
    x_u: jax.Array,
    time_fractions: jax.Array,
    running_cost_integrator: Any,
    running_cost_func: Callable[[jax.Array, jax.Array, jax.Array], float],
    terminal_cost_func: Callable[[float, jax.Array, float, jax.Array], float],
    dynamics: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    x_shape: Tuple[int, int],
    u_shape: Tuple[int, int],
):
    running_cost = running_cost_integrator(
        x_u, time_fractions, running_cost_func, dynamics, x_shape, u_shape
    )
    x, _, t_0, t_f = _unpack(x_u, x_shape, u_shape)
    terminal_cost = terminal_cost_func(t_0, x[0], t_f, x[-1])
    return running_cost + terminal_cost


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _collocation_and_path_constraints(
    x_u: jax.Array,
    time_fractions: jax.Array,
    collocation_constraints: Callable[[jax.Array], jax.Array],
    path_constraints: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    x_shape: Tuple[int, int],
    u_shape: Tuple[int, int],
):
    x, u, t_0, t_f = _unpack(x_u, x_shape, u_shape)
    t = _get_time(t_0, t_f, time_fractions)
    return jnp.concatenate(
        [collocation_constraints(x_u), path_constraints(x, u, t).reshape(-1)]
    )


class OCP:
    _METHOD_ALIASES = {
        "trapezoidal": trapezoidal.TrapezoidalTrajectory,
        "hermite-simpson": hermite_simpson.HermiteSimpsonTrajectory,
    }

    def __init__(
        self,
        dynamics: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        t_0_bounds: Tuple[float, float],
        t_f_bounds: Tuple[float, float],
        x_0_bounds: Tuple[jax.Array, jax.Array],
        x_f_bounds: Tuple[jax.Array, jax.Array],
        x_bounds: Tuple[jax.Array, jax.Array],
        u_bounds: Tuple[jax.Array, jax.Array],
        initial_guess: Guess,
        n_grid: int,
        method: str = "trapezoidal",
        path_constraints: Optional[
            Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
        ] = None,
        path_constraints_bounds: Optional[Tuple[jax.Array, jax.Array]] = None,
        running_cost: Optional[
            Callable[[jax.Array, jax.Array, jax.Array], float]
        ] = None,
        terminal_cost: Optional[
            Callable[[float, jax.Array, float, jax.Array], float]
        ] = None,
        max_mesh_refinement_iters: Optional[int] = None,
        error_tolerance: Optional[float] = None,
        predicted_error_kappa: float = 0.1,
        max_new_points_per_interval: int = 5,
        solver_kwargs: Optional[dict] = None,
        jac_sparsity_estimation_samples: int = 100,
    ):
        if method not in self._METHOD_ALIASES:
            raise ValueError(
                f"Choose one collocation method from {list(self._METHOD_ALIASES.keys())}"
            )
        self._set_collocation_method(method)
        self._dynamics = jax.jit(jax.vmap(dynamics, in_axes=(0, 0, 0)))
        if path_constraints is not None and path_constraints_bounds is None:
            raise ValueError(
                "Provide bounds for the path constraints (path_constraints_bounds)"
            )
        self._path_constraints = (
            jax.jit(jax.vmap(path_constraints, in_axes=(0, 0, 0)))
            if path_constraints is not None
            else None
        )
        if path_constraints is not None and path_constraints_bounds is not None:
            (
                self._path_constraints_lower,
                self._path_constraints_upper,
            ) = path_constraints_bounds
        running_cost = (lambda *_: 0) if running_cost is None else running_cost
        self._running_cost = jax.vmap(running_cost, in_axes=(0, 0, 0))
        const_zero_func: Callable[
            [float, jax.Array, float, jax.Array], float
        ] = lambda *_: 0.0
        self._terminal_cost = jax.jit(
            const_zero_func if terminal_cost is None else terminal_cost
        )
        self._t_0_lower, self._t_0_upper = t_0_bounds
        self._t_f_lower, self._t_f_upper = t_f_bounds
        self._x_0_lower, self._x_0_upper = x_0_bounds
        self._x_f_lower, self._x_f_upper = x_f_bounds
        self._x_lower, self._x_upper = x_bounds
        self._u_lower, self._u_upper = u_bounds
        self._time_fractions = jnp.linspace(0, 1, n_grid)
        self._initial_guess = initial_guess
        self._max_mesh_refinement_iters = max_mesh_refinement_iters
        self._error_tolerance = error_tolerance
        self._predicted_error_kappa = predicted_error_kappa
        self._max_new_points_per_interval = max_new_points_per_interval
        self._solver_kwargs = solver_kwargs
        self._old_discretization_errors = np.zeros(n_grid)
        self._jac_sparsity_estimation_samples = jac_sparsity_estimation_samples

    def _set_collocation_method(self, method_name: str):
        self._trajectory_subclass = self._METHOD_ALIASES[method_name]

    def _estimate_jacobian_structure(
        self,
        n_samples: int,
        initial_x_u: jax.Array,
        constraints: Callable[[jax.Array], jax.Array],
        seed: int = 42,
    ) -> Tuple[jax.Array, ...]:
        key = jax.random.key(seed)
        key, *subkeys = jax.random.split(key, n_samples)
        get_jacobian = jax.jit(jax.jacobian(constraints))
        jac_sum = np.zeros(get_jacobian(initial_x_u).shape)
        for subkey in subkeys:
            perturbation = jax.random.normal(subkey, shape=initial_x_u.shape)
            jacobian = get_jacobian(initial_x_u + perturbation)
            jac_sum += jnp.abs(jacobian)
        return jnp.nonzero(jac_sum)

    def _pack_initial_guess(self) -> jax.Array:
        guess_t_0, guess_t_f = (
            self._initial_guess._t.min(),
            self._initial_guess._t.max(),
        )
        return _pack(
            *self._initial_guess.interpolate(
                _get_time(guess_t_0, guess_t_f, self._time_fractions),
                self._trajectory_subclass._USES_U_MIDPOINTS,
            ),
            guess_t_0,
            guess_t_f,
        )

    def _get_x_u_shapes(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        n_grid = self._time_fractions.size
        x_shape = (n_grid, self._x_0_lower.size)
        u_shape = (
            n_grid
            if not self._trajectory_subclass._USES_U_MIDPOINTS
            else 2 * n_grid - 1,
            self._u_lower.size,
        )
        return x_shape, u_shape

    def _build_nlp(self, packed_initial_guess: jax.Array) -> cyipopt.Problem:
        n_grid = self._time_fractions.size
        x_shape, u_shape = self._get_x_u_shapes()
        problem = types.SimpleNamespace()
        problem.objective = lambda x_u: _objective(
            x_u,
            self._time_fractions,
            self._trajectory_subclass._objective,
            self._running_cost,
            self._terminal_cost,
            self._dynamics,
            x_shape,
            u_shape,
        )
        problem.gradient = jax.jit(jax.grad(problem.objective))
        collocation_constraints = (
            lambda x_u: self._trajectory_subclass._collocation_constraints(
                x_u, self._time_fractions, self._dynamics, x_shape, u_shape
            )
        )
        num_collocation_constraints = collocation_constraints(packed_initial_guess).size
        constraints = (
            collocation_constraints
            if self._path_constraints is None
            else lambda x_u: _collocation_and_path_constraints(
                x_u,
                self._time_fractions,
                collocation_constraints,
                self._path_constraints,
                x_shape,
                u_shape,
            )
        )
        problem.constraints = constraints
        jacobian_structure = self._estimate_jacobian_structure(
            self._jac_sparsity_estimation_samples, packed_initial_guess, constraints
        )
        problem.jacobianstructure = jax.jit(lambda: jacobian_structure)
        problem.jacobian = jax.jit(
            lambda x_u: jax.jacobian(constraints)(x_u)[jacobian_structure]
        )
        num_variables = packed_initial_guess.size
        num_constraints = constraints(packed_initial_guess).size
        lb_x, ub_x = [
            jnp.tile(arr, (n_grid, 1)) for arr in [self._x_lower, self._x_upper]
        ]
        lb_u, ub_u = [
            jnp.tile(
                arr,
                (
                    n_grid
                    if not self._trajectory_subclass._USES_U_MIDPOINTS
                    else 2 * n_grid - 1,
                    1,
                ),
            )
            for arr in [self._u_lower, self._u_upper]
        ]
        lb_x = lb_x.at[0].set(self._x_0_lower).at[-1].set(self._x_f_lower)
        ub_x = ub_x.at[0].set(self._x_0_upper).at[-1].set(self._x_f_upper)
        lb, ub = _pack(lb_x, lb_u, self._t_0_lower, self._t_f_lower), _pack(
            ub_x, ub_u, self._t_0_upper, self._t_f_upper
        )
        constraint_bounds = (
            2 * (jnp.zeros(num_collocation_constraints),)
            if self._path_constraints is None
            else [
                jnp.concatenate(
                    [
                        jnp.zeros(num_collocation_constraints),
                        jnp.tile(jnp.asarray(path_constraints_bound), n_grid),
                    ]
                )
                for path_constraints_bound in [
                    self._path_constraints_lower,
                    self._path_constraints_upper,
                ]
            ]
        )
        problem = cyipopt.Problem(
            problem_obj=problem,
            n=num_variables,
            m=num_constraints,
            cl=constraint_bounds[0],
            cu=constraint_bounds[1],
            lb=lb,
            ub=ub,
        )
        if self._solver_kwargs is None:
            return problem
        for key, value in self._solver_kwargs.items():
            problem.add_option(key, value)
        return problem

    def solve(self) -> Tuple[Trajectory, float]:
        i = 0
        while True:
            if (
                self._max_mesh_refinement_iters is not None
                and i >= self._max_mesh_refinement_iters
            ):
                break
            logging.getLogger(__name__).info(
                f"Number of grid points is {self._time_fractions.size}."
            )
            initial_x_u = self._pack_initial_guess()
            nlp = self._build_nlp(initial_x_u)
            x_u, info = nlp.solve(initial_x_u)
            nlp.close()
            x_shape, u_shape = self._get_x_u_shapes()
            x, u, t_0, t_f = _unpack(x_u, x_shape, u_shape)
            t = _get_time(t_0, t_f, self._time_fractions)
            trajectory = self._trajectory_subclass(t, x, u, self._dynamics)
            self._initial_guess = Guess.from_trajectory(trajectory)
            if self._error_tolerance is None:
                break
            errors = self._get_discretization_errors(x, u, t, trajectory)
            logging.getLogger(__name__).info(
                f"Maximum discretization error is {errors.max()}."
            )
            if errors.max() <= self._error_tolerance:
                break
            logging.getLogger(__name__).info(f"Remeshing trajectory.")
            self._remesh_trajectory(errors, i)
            i += 1
        return trajectory, info["obj_val"]

    def _remesh_trajectory(self, jnp_errors: jax.Array, iteration: int):
        assert self._error_tolerance is not None
        errors = np.array(jnp_errors)
        self._set_new_mesh_order(errors, iteration)
        added_points = np.zeros(self._time_fractions.size - 1).astype(int)
        order_reductions = self._estimate_order_reductions(
            self._old_discretization_errors, errors
        )
        self._old_discretization_errors = errors
        while True:
            max_error_pos = errors.argmax()
            max_error = errors[max_error_pos]
            total_added_points = added_points.sum()
            if total_added_points >= min(
                self._max_new_points_per_interval,
                round(self._predicted_error_kappa * self._time_fractions.size),
            ) and (
                max_error <= self._error_tolerance
                and added_points[max_error_pos] == 0
                or max_error <= self._error_tolerance * self._predicted_error_kappa
                and added_points[max_error_pos] > 0
                and added_points[max_error_pos] < self._max_new_points_per_interval
                or total_added_points >= self._time_fractions.size - 1
                or added_points.max() >= self._max_new_points_per_interval
            ):
                break
            added_points[max_error_pos] += 1
            errors[max_error_pos] *= (1 / (1 + added_points[max_error_pos])) ** (
                self._trajectory_subclass._ORDER - order_reductions[max_error_pos] + 1
            )
        self._old_added_points = added_points
        new_time_fractions = []
        for i, n_points in enumerate(added_points):
            t_a, t_b = self._time_fractions[jnp.asarray([i, i + 1])]
            new_time_fractions.append(jnp.linspace(t_a, t_b, n_points + 2)[:-1])
        new_time_fractions.append(jnp.array([self._time_fractions[-1]]))
        self._time_fractions = jnp.concatenate(new_time_fractions)

    def _set_new_mesh_order(self, errors: np.ndarray, iteration: int):
        if self._trajectory_subclass._ORDER >= 4:
            return
        error_is_equidistributed = errors.max() <= 2 * errors.mean()
        if error_is_equidistributed or iteration >= 2:
            self._set_collocation_method("hermite-simpson")
            logging.getLogger(__name__).info(
                f"Switching to Hermite-Simpson collocation."
            )

    def _get_discretization_errors(
        self, x: jax.Array, u: jax.Array, t: jax.Array, trajectory: Trajectory
    ) -> jax.Array:
        variable_weights = (
            jnp.abs(
                jnp.vstack(
                    [
                        self._dynamics(
                            x,
                            u
                            if not self._trajectory_subclass._USES_U_MIDPOINTS
                            else u[::2],
                            t.reshape(-1, 1),
                        ),
                        x,
                    ]
                )
            )
            .max(axis=0)
            .reshape(-1)
        )
        errors = scipy.integrate.quad_vec(
            lambda t_frac: _collocation_error(t_frac, t, self, trajectory),
            0,
            1,
        )[0] * (t[1:] - t[:-1]).reshape(-1, 1)
        return (errors / (variable_weights + 1)).max(axis=1)

    def _estimate_order_reductions(
        self, old_errors: np.ndarray, errors: np.ndarray
    ) -> jax.Array:
        if jnp.isclose(old_errors.max(), 0):
            return jnp.zeros(errors.size)
        new_error_indices = jnp.insert(self._old_added_points + 1, 0, 0).cumsum()[:-1]
        new_errors = errors[new_error_indices]
        r_hat = (
            self._trajectory_subclass._ORDER
            + 1
            - jnp.log(old_errors / new_errors) / jnp.log(self._old_added_points + 1)
        )
        return jnp.maximum(
            0, jnp.minimum(jnp.round(r_hat), self._trajectory_subclass._ORDER)
        )


def _collocation_error(t_frac: jax.Array, t: jax.Array, ocp: OCP, solution: Trajectory):
    t = t[:-1] + t_frac * (t[1:] - t[:-1])
    x, u = solution.interpolate(t)
    return jnp.abs((ocp._dynamics(x, u, t) - solution._approx_dynamics(t)))
