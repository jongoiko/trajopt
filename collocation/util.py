from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _pack(x: jax.Array, u: jax.Array, t_0: float, t_f: float) -> jax.Array:
    return jnp.concatenate(
        [x.reshape(-1), u.reshape(-1), jnp.array([t_0]), jnp.array([t_f])]
    )


@partial(jax.jit, static_argnums=(1, 2))
def _unpack(
    x_u: jax.Array, x_shape: Tuple[int, int], u_shape: Tuple[int, int]
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    num_x_vars = np.prod(np.asarray(x_shape))
    return (
        x_u[:num_x_vars].reshape(*x_shape),
        x_u[num_x_vars:-2].reshape(*u_shape),
        x_u[-2],
        x_u[-1],
    )


@jax.jit
def _get_time(t_0: float, t_f: float, time_fractions: jax.Array) -> jax.Array:
    return t_0 + time_fractions * (t_f - t_0)


_lerp = jax.jit(jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1))
