
.. _nonlinear_least_squares:

Nonlinear least squares
=======================

This section is concerned with problems of the form

.. math::

    \min_{x} \frac{1}{2} ||\textbf{r}(x, \theta)||^2,

where :math:`\textbf{r}` is is a residual function, :math:`x` are the
parameters with respect to which the function is minimized, and :math:`\theta`
are optional additional arguments.

Gauss-Newton
------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.GaussNewton

We can use the Gauss-Newton method, which is the standard approach for nonlinear least squares problems.

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

As an example, let us see how to minimize the Rosenbrock residual function::

  from jaxopt import GaussNewton

  def rosenbrock(x):
      return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])

  gn = GaussNewton(residual_fun=rosenbrock)
  gn_sol = gn.run(x_init).params


The residual function may take additional arguments, for example for fitting a double exponential::

  def double_exponential(x, x_data, y_data):
    return y_data - (x[0] * jnp.exp(-x[2] * x_data) + x[1] * jnp.exp(
        -x[3] * x_data)).

  gn = GaussNewton(residual_fun=double_exponential)
  gn_sol = gn.run(x_init, x_data=x_data, y_data=y_data).params

Differentiation
~~~~~~~~~~~~~~~

In some applications, it is useful to differentiate the solution of the solver
with respect to some hyperparameters.  Continuing the previous example, we can
now differentiate the solution w.r.t. ``y``::

  def solution(y):
    gn = GaussNewton(residual_fun=double_exponential)
    lm_sol = lm.run(x_init, x_data, y).params

  print(jax.jacobian(solution)(y_data))

Under the hood, we use the implicit function theorem if ``implicit_diff=True``
and autodiff of unrolled iterations if ``implicit_diff=False``.  See the
:ref:`implicit differentiation <implicit_diff>` section for more details.

Levenberg Marquardt
-------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.LevenbergMarquardt

We can also use the Levenberg-Marquardt method, which is a more advanced method compared to Gauss-Newton, in
that it regularizes the update equation. It helps for cases where Gauss-Newton method fails to converge.

Update equation
~~~~~~~~~~~~~~~

The following equation is solved for every iteration to find the update to the
parameters:

.. math::
    (\mathbf{J} \mathbf{J^T} + \mu\mathbf{I}) h_{lm} = - \mathbf{J^T} \mathbf{r}

where :math:`\mathbf{J}` is the Jacobian of the residual function w.r.t.
parameters and :math:`\mu` is the damping parameter.
