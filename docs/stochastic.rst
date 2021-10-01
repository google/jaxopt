Stochastic optimization
=======================

This section is concerned with problems of the form

.. math::

    \min_{x} \mathbb{E}_{D}[f(x, \theta, D)],

where :math:`f(x, \theta, D)` is differentiable (almost everywhere), :math:`x`
are the parameters with respect to which the function is minimized,
:math:`\theta` are optional fixed extra arguments and :math:`D` is a random
variable (typically a mini-batch).


.. topic:: Examples

   * :ref:`sphx_glr_auto_examples_deep_learning_haiku_mnist.py`
   * :ref:`sphx_glr_auto_examples_deep_learning_flax_mnist.py`
   * :ref:`sphx_glr_auto_examples_deep_learning_haiku_vae.py`


Defining an objective function
------------------------------

Objective functions must contain a ``data`` argument corresponding to :math:`D` above.

Example::

  def ridge_reg_objective(params, l2reg, data):
    X, y = data
    residuals = jnp.dot(X, params) - y
    return jnp.mean(residuals ** 2) + 0.5 * l2reg * jnp.dot(w ** 2)

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

Optax solvers
~~~~~~~~~~~~~

`Optax <https://optax.readthedocs.io>`_ solvers can be used in JAXopt using
:class:`OptaxSolver <jaxopt.OptaxSolver>`. Here's an example with Adam::

    from jaxopt import OptaxSolver

    opt = optax.adam(learning_rate)
    solver = OptaxSolver(opt=opt, fun=ridge_reg_objective, maxiter=1000)

See `common optimizers
<https://optax.readthedocs.io/en/latest/api.html#common-optimizers>`_ in the
optax documentation for a list of available stochastic solvers.

Adaptive solvers
~~~~~~~~~~~~~~~~

Adaptive solvers update the step size at each iteration dynamically.
An example is :class:`PolyakSGD <jaxopt.PolyakSGD>`, a solver
which computes step sizes adaptively using function values.

Run iterator vs. manual loop
----------------------------

The following::

  iterator = data_iterator()
  solver.run_iterator(init_params, iterator, l2reg=l2reg)

is equivalent to::

  iterator = data_iterator()
  params, state = solver.init(init_params, l2reg=l2reg)
  for _ in range(maxiter):
    data = next(iterator)
    params, state = solver.update(params, state, l2reg=l2reg, data=data)
