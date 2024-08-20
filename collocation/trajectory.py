from typing import Tuple, Callable
import jax
import abc


class Trajectory(abc.ABC):
    _ORDER: int = NotImplemented
    _USES_U_MIDPOINTS: bool = NotImplemented

    def __init__(self, t: jax.Array, x: jax.Array, u: jax.Array, x_dot: jax.Array):
        self.t_0, self.t_f = t.min(), t.max()
        self._t = t
        self._x = x
        self._u = u
        self._x_dot = x_dot

    @abc.abstractmethod
    def interpolate(self, t: jax.Array) -> Tuple[jax.Array, jax.Array]:
        pass

    @abc.abstractmethod
    def _approx_dynamics(self, t: jax.Array) -> jax.Array:
        pass

    @staticmethod
    @abc.abstractmethod
    def _objective(
        x_u: jax.Array,
        time_fractions: jax.Array,
        running_cost: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        dynamics: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        x_shape: Tuple[int, int],
        u_shape: Tuple[int, int],
    ) -> float:
        pass

    @staticmethod
    @abc.abstractmethod
    def _collocation_constraints(
        x_u: jax.Array,
        time_fractions: jax.Array,
        dynamics: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        x_shape: Tuple[int, int],
        u_shape: Tuple[int, int],
    ) -> jax.Array:
        pass
