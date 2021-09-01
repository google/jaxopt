Unconstrained optimization
==========================

This section is concerned with problems of the form

.. math::

    \min_{x} f(x, \theta)

where :math:`f(x, \theta)` is a differentiable (almost everywhere), :math:`x`
are the parameters with respect to which the function is minimized and
:math:`\theta` are optional extra arguments.

Defining an objective function
------------------------------

Objective functions must always include as first argument the variable with
respect to which the function is minimized. The function can also contain extra
arguments.

The following illustrates how to express the ridge regression objective::

  def ridge_reg_objective(params, l2reg, X, y):
    residuals = jnp.dot(X, params) - y
    return jnp.mean(residuals) + 0.5 * l2reg * jnp.dot(w ** 2)

The model parameters ``params`` correspond to :math:`x` while ``l2reg``, ``X``
and ``y`` correspond to the extra arguments :math:`\theta` in the mathematical
notation above.

Solvers
-------

.. autosummary::
  :toctree: _autosummary

    jaxopt.GradientDescent
    jaxopt.ScipyMinimize

Continuing the ridge regression example above, gradient descent can be
instantiated and run as follows::

  gd = jaxpot.GradientDescent(fun=ridge_reg_objective, maxiter=500)
  params = gd.run(init_params, l2reg=l2reg, X=X, y=y).params
