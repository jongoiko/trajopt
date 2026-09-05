# Trajectory optimization with direct collocation

This is a Python implementation of a trajectory optimizer based on direct collocation [[1]](#1), using [Jax](https://docs.jax.dev/en/latest/) for automatic differentiation and [IPOPT](https://coin-or.github.io/Ipopt/) for optimization.
It supports trapezoidal and Hermite-Simpson collocation, as well as an experimental implementation of Betts' mesh refinement method [[2]](#2).

## What is direct collocation?

We aim to solve a continuous-time optimal control problem (OCP) defined by

$$
\min_{x(t),\,u(t)} \int_{t_0}^{t_f} L(x(t), u(t), t)\, \text{d}t + L_f(x(t_0), x(t_f), t_0, t_f)
$$

subject to

$$
\dot{x}(t) = f(x(t), u(t), t),
$$

where $x(t)$ is the state, $u(t)$ the control, $f$ the system dynamics, $L$ the running cost and $L_f$ is the terminal cost.
We can also impose additional constraints on the whole trajectory and/or initial/final states.

Direct collocation converts the continuous-time optimal control problem into a finite-dimensional nonlinear program (NLP) by discretizing the state and control trajectories at a set of grid points, then enforcing the system dynamics only at those points.
We discretize the time interval $[t_0, t_f]$ into $N-1$ intervals over a grid of $N$ points, placed at fixed fractions of the horizon so that $t_k = t_0 + \tau_k (t_f - t_0)$.
The states and controls at the grid points, $x_k \approx x(t_k)$ and $u_k \approx u(t_k)$, together with the free endpoint times $t_0, t_f$, become the NLP decision variables.

Writing $h_k = t_{k+1} - t_k$, instead of integrating $f$ forward as in [shooting methods](https://en.wikipedia.org/wiki/Shooting_method), collocation enforces the ODE only implicitly through _defect constraints_, requiring the discretized trajectory to satisfy system dynamics at each segment.

### Collocation methods

Here we implement _trapezoidal_ and _Hermite-Simpson_ collocation.

#### Trapezoidal collocation

Trapezoidal collocation uses only the grid points $x_k, u_k, t_k$, with $\dot{x}_k := f(x_k, u_k, t_k)$. The defect constraints are
$$
x_{k+1} - x_k - \frac{h_k}{2}\Big(\dot{x}_{k+1} + \dot{x}_k\Big) = 0,
$$
and the integrated running cost is approximated by
$$
\sum_{k=0}^{N-2} \frac{h_k}{2}\Big(L(x_k,u_k,t_k) + L(x_{k+1},u_{k+1},t_{k+1})\Big).
$$

#### Hermite–Simpson collocation

Hermite–Simpson adds a midpoint control $u_{k+\frac 1 2}$ as an extra decision variable, and a midpoint state $x_{k+\frac 1 2}$ computed from a cubic Hermite fit, thus given by
$$
x_{k+\frac 1 2} = \frac{1}{2}(x_k + x_{k+1}) + \frac{h_k}{8}\Big(\dot{x}_k - \dot{x}_{k+1}\Big),
$$
with $t_{k+\frac 1 2} = t_k + h_k/2$ and $\dot{x}_{k+\frac 1 2} := f(x_{k+\frac 1 2}, u_{k+\frac 1 2}, t_{k+\frac 1 2})$.

The defect constraints are
$$
x_{k+1} - x_k - \frac{h_k}{6}\Big(\dot{x}_{k+1} + 4\dot{x}_{k+\frac 1 2} + \dot{x}_k\Big) = 0,
$$
and we approximate the running cost by
$$
\sum_{k=0}^{N-2} \frac{h_k}{6}\Big(L(x_k,u_k,t_k) + 4L(x_{k+\frac 1 2},u_{k+\frac 1 2},t_{k+\frac 1 2}) + L(x_{k+1},u_{k+1},t_{k+1})\Big).
$$

## Mesh refinement

Because accuracy depends on $h_k$, a fixed grid can be too coarse in some intervals and too fine in others.
The OCP solving routine thus estimates a per-interval discretization error, comparing the true dynamics $f$ against the trajectory's local polynomial approximation.
Then, following Betts' method, mesh refinement either raises the collocation order (from trapezoidal to Hermite–Simpson) or adds grid points in the highest-error intervals, re-solving until the maximum error is below a certain tolerance or a maximum number of iterations is reached.

---

## Resources and references

<a id="1">[1]</a>
Kelly, Matthew. "An introduction to trajectory optimization: How to do your own direct collocation." SIAM review 59.4 (2017): 849-904.

<a id="2">[2]</a>
Betts, John T. Practical methods for optimal control and estimation using nonlinear programming. Society for Industrial and Applied Mathematics, 2010.
