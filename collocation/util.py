from functools import partial
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
