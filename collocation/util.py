from functools import partial
import numpy as np
import jax
import jax.numpy as jnp


@jax.jit
def _pack(x, u, t_0, t_f):
    return jnp.concatenate(
        [x.reshape(-1), u.reshape(-1), jnp.array([t_0]), jnp.array([t_f])]
    )


@partial(jax.jit, static_argnums=(1, 2))
def _unpack(x_u, x_shape, u_shape):
    num_x_vars = np.prod(np.asarray(x_shape))
    return (
        x_u[:num_x_vars].reshape(*x_shape),
        x_u[num_x_vars:-2].reshape(*u_shape),
        x_u[-2],
        x_u[-1],
    )


@jax.jit
def _get_time(t_0, t_f, time_fractions):
    return t_0 + time_fractions * (t_f - t_0)


_lerp = jax.jit(jax.vmap(jnp.interp, in_axes=(None, None, 1)))
