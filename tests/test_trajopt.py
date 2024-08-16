import pytest
import trajopt
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")


@pytest.fixture(scope="session")
def min_force_squared_point_mass():
    def dynamics(x, u, t):
        return jnp.array([x[1], u[0]])

    def cost(x, u, t):
        return u[0] ** 2

    t_0, t_f = 0, 1
    x_0, x_f = jnp.array([0, 0]), jnp.array([1, 0])

    guess = trajopt.Guess(
        jnp.array([t_0, t_f]),
        jnp.vstack([x_0, x_f]),
        jnp.array([[-1], [1]]),
    )

    x_lower = jnp.array([0, -jnp.inf])
    x_upper = jnp.array([1, jnp.inf])
    u_lower = jnp.array([-50])
    u_upper = -u_lower

    n_grid = 30

    ocp = trajopt.OCP(
        dynamics,
        (t_0, t_0),
        (t_f, t_f),
        (x_0, x_0),
        (x_f, x_f),
        (x_lower, x_upper),
        (u_lower, u_upper),
        guess,
        n_grid,
        running_cost=cost,
        method="trapezoidal",
    )

    return ocp.solve()


@pytest.fixture(scope="session")
def min_force_squared_cart_pole():
    G = 9.81
    CART_MASS = 2.0
    POLE_MASS = 0.5
    L = 0.5

    DIST = 0.8
    MAX_FORCE = 100

    def dynamics(x, u, t):
        _, vel, theta, theta_dot = x
        f = u[0]
        theta_ddot = -(
            L * POLE_MASS * jnp.cos(theta) * jnp.sin(theta) * theta_dot * theta_dot
            + f * jnp.cos(theta)
            + (CART_MASS + POLE_MASS) * G * jnp.sin(theta)
        ) / (L * CART_MASS + L * POLE_MASS * (1 - jnp.cos(theta) * jnp.cos(theta)))
        vel_dot = (
            L * POLE_MASS * jnp.sin(theta) * theta_dot * theta_dot
            + f
            + POLE_MASS * G * jnp.cos(theta) * jnp.sin(theta)
        ) / (CART_MASS + POLE_MASS * (1 - jnp.cos(theta) * jnp.cos(theta)))
        return jnp.array([vel, vel_dot, theta_dot, theta_ddot])

    def cost(x, u, t):
        return u[0] ** 2

    t_0, t_f = 0, 2
    x_0, x_f = jnp.array([0, 0, 0, 0]), jnp.array([DIST, 0, jnp.pi, 0])

    guess = trajopt.Guess(
        jnp.array([t_0, t_f]),
        jnp.vstack([x_0, x_f]),
        jnp.array([[0], [0]]),
    )

    x_lower = jnp.array([-2 * DIST, -jnp.inf, -2 * jnp.pi, -jnp.inf])
    x_upper = -x_lower
    u_lower = jnp.array([-MAX_FORCE])
    u_upper = -u_lower

    n_grid = 30

    ocp = trajopt.OCP(
        dynamics,
        (t_0, t_0),
        (t_f, t_f),
        (x_0, x_0),
        (x_f, x_f),
        (x_lower, x_upper),
        (u_lower, u_upper),
        guess,
        n_grid,
        method="trapezoidal",
        running_cost=cost,
    )

    return ocp.solve()


def test_min_force_squared_point_mass_objective(min_force_squared_point_mass):
    _, objective = min_force_squared_point_mass
    assert jnp.isclose(objective, 12.055857640099498)


def test_min_force_squared_cart_pole_objective(min_force_squared_cart_pole):
    _, objective = min_force_squared_cart_pole
    assert jnp.isclose(objective, 229.2912807061593)
