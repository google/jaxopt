Stochastic optimization
=======================

This section is concerned with problems of the form

.. math::

    \min_{x} \mathbb{E}_{D}[f(x, \theta, D)],

where :math:`f(x, \theta, D)` is differentiable (almost everywhere), :math:`x`
are the parameters with respect to which the function is minimized,
:math:`\theta` are optional fixed extra arguments and :math:`D` is a random
variable (typically a mini-batch).

Defining an objective function
------------------------------

Objective functions must contain a ``data`` argument corresponding to :math:`D` above.

Example::

  def ridge_reg_objective(params, l2reg, data):
    X, y = data
    residuals = jnp.dot(X, params) - y
    return jnp.mean(residuals) + 0.5 * l2reg * jnp.dot(w ** 2)

Data iterator
-------------

Sampling realizations of the random variable :math:`D` can be done using an iterator.

Example::

  def data_iterator():
    for _ in range(n_iter):
      perm = rng.permutation(n_samples)[:batch_size]
      yield (X[perm], y[perm])

Solvers
-------

.. autosummary::
  :toctree: _autosummary

    jaxopt.OptaxSolver
    jaxopt.PolyakSGD

Example::

    opt = optax.adam(learning_rate)
    solver = OptaxSolver(opt=opt, fun=ridge_reg_objective, maxiter=1000)

Run iterator vs. manual loop
----------------------------

The following::

  solver.run_iterator(init_params, iterator, l2reg=l2reg)

is equivalent to::

  params, state = solver.init(init_params, l2reg=l2reg)
  for _ in range(maxiter):
    data = next(iterator)
    params, state = solver.update(params, state, l2reg=l2reg, data=data)
