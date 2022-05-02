
.. _nonlinear_least_squares:

Least squares optimization
==========================

This section is concerned with problems of the form

.. math::

    \min_{x} f(x) = \frac{1}{2} * ||\textbf{r}(x, \theta)||^2=\sum_{i=1}^m r_i(x_1,...,x_n)^2,

where :math:`r \colon \mathbb{R}^n \to \mathbb{R}^m` is :math:`r(x, \theta)` is
:math:`r(x, \theta)` is differentiable (almost everywhere), :math:`x` are the
parameters with respect to which the function is minimized, and :math:`\theta`
are optional additional arguments.

Gauss Newton
------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.GaussNewton

Update equation
~~~~~~~~~~~~~~~

The following equation is solved for every iteration to find the update to the
parameters:

.. math::
    \mathbf{J} \mathbf{J^T} h_{gn} = - \mathbf{J^T} \mathbf{r}

where :math:`\mathbf{J}` is the Jacobian of the residual function w.r.t.
parameters.

Instantiating and running the solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To solve nonlinear least squares optimization problems, we can use Gauss Newton
method, which is the standard approach for nonlinear least squares problems ::

  from jaxopt import GaussNewton

  gn = GaussNewton(residual_fun=fun)
  gn_sol = gn.run(x_init, *args, **kwargs).params

As an example, consider the Rosenbrock residual function ::

  def rosenbrock_res_fun(x):
      return np.array([10 * (x[1] - x[0]**2), (1 - x[0])]).

The function can take arguments, for example for fitting a double exponential ::

  def double_exponential_fit(x, x_data, y_data):
    return y_data - (x[0] * jnp.exp(-x[2] * x_data) + x[1] * jnp.exp(
        -x[3] * x_data)).

Differentiation
~~~~~~~~~~~~~~~

In some applications, it is useful to differentiate the solution of the solver
with respect to some hyperparameters.  Continuing the previous example, we can
now differentiate the solution w.r.t. ``y``::

  def solution(y):
    gn = GaussNewton(residual_fun=fun)
    lm_sol = lm.run(x_init, X, y).params

  print(jax.jacobian(solution)(y))

Under the hood, we use the implicit function theorem if ``implicit_diff=True``
and autodiff of unrolled iterations if ``implicit_diff=False``.  See the
:ref:`implicit differentiation <implicit_diff>` section for more details.

Levenberg Marquardt
-------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.LevenbergMarquardt

Update equation
~~~~~~~~~~~~~~~

The following equation is solved for every iteration to find the update to the
parameters:

.. math::
    (\mathbf{J} \mathbf{J^T} + \mu\mathbf{I}) h_{lm} = - \mathbf{J^T} \mathbf{r}

where :math:`\mathbf{J}` is the Jacobian of the residual function w.r.t.
parameters and :math:`\mu` is the damping parameter..

Instantiating and running the solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To solve nonlinear least squares optimization problems, we can use Levenberg
Marquardt method, which is a more advanced method compared to Gauss Newton, in
that it regularizes the update equation which helps for cases where Gauss
Newton method fails to converge ::

  from jaxopt import LevenbergMarquardt

  lm = LevenbergMarquardt(residual_fun=fun)
  lm_sol = lm.run(x_init, X, y).params
