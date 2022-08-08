.. _unconstrained_optim:

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

Objective functions must always include as first argument the variables with
respect to which the function is minimized. The function can also contain extra
arguments.

The following illustrates how to express the ridge regression objective::

  def ridge_reg_objective(params, l2reg, X, y):
    residuals = jnp.dot(X, params) - y
    return jnp.mean(residuals ** 2) + 0.5 * l2reg * jnp.sum(params ** 2)

The model parameters ``params`` correspond to :math:`x` while ``l2reg``, ``X``
and ``y`` correspond to the extra arguments :math:`\theta` in the mathematical
notation above.

Solvers
-------

.. autosummary::
  :toctree: _autosummary

    jaxopt.GradientDescent
    jaxopt.LBFGS
    jaxopt.ScipyMinimize
    jaxopt.NonlinearCG

Instantiating and running the solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continuing the ridge regression example above, gradient descent can be
instantiated and run as follows::

  solver = jaxopt.LBFGS(fun=ridge_reg_objective, maxiter=maxiter)
  res = solver.run(init_params, l2reg=l2reg, X=X, y=y)

  # Alternatively, we could have used one of these solvers as well:
  # solver = jaxopt.GradientDescent(fun=ridge_reg_objective, maxiter=500)
  # solver = jaxopt.ScipyMinimize(fun=ridge_reg_objective, method="L-BFGS-B")
  # solver = jaxopt.NonlinearCG(fun=ridge_reg_objective, method="polak-ribiere", maxiter=500)

Unpacking results
~~~~~~~~~~~~~~~~~

Note that ``res`` has the form ``NamedTuple(params, state)``, where ``params``
are the approximate solution found by the solver and ``state`` contains
solver-specific information about convergence.

Because ``res`` is a ``NamedTuple``, we can unpack it as::

  params, state = res
  print(params, state)

Alternatively, we can also access attributes directly::

  print(res.params, res.state)
