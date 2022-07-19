Non-smooth optimization
=======================

This section is concerned with problems of the form

.. math::

    \min_{x} f(x, \theta) + g(x, \lambda)

where :math:`f(x, \theta)` is differentiable (almost everywhere),
:math:`x` are the parameters with respect to which the function is minimized,
:math:`\theta` are optional extra arguments,
:math:`g(x, \lambda)` is possibly non-smooth,
and :math:`\lambda` are extra parameters :math:`g` may depend on.


Proximal gradient
-----------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.ProximalGradient

Instantiating and running the solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Proximal gradient is a generalization of :ref:`projected gradient descent
<constrained_optim>`. The non-smooth term :math:`g` above is specified by
setting the corresponding proximal operator, which is achieved using the
``prox`` attribute of :class:`ProximalGradient <jaxopt.ProximalGradient>`.

For instance, suppose we want to solve the following optimization problem

.. math::

    \min_{w} \frac{1}{2n} ||Xw - y||^2 + \text{l1reg} \cdot ||w||_1

which corresponds to the choice :math:`g(w, \text{l1reg}) = \text{l1reg} \cdot ||w||_1`.  The
corresponding ``prox`` operator is :func:`prox_lasso <jaxopt.prox.prox_lasso>`.
We can therefore write::

  from jaxopt import ProximalGradient
  from jaxopt.prox import prox_lasso

  def least_squares(w, data):
    X, y = data
    residuals = jnp.dot(X, w) - y
    return jnp.mean(residuals ** 2)

  l1reg = 1.0
  pg = ProximalGradient(fun=least_squares, prox=prox_lasso)
  pg_sol = pg.run(w_init, hyperparams_prox=l1reg, data=(X, y)).params

Note that :func:`prox_lasso <jaxopt.prox.prox_lasso>` has a hyperparameter
``l1reg``, which controls the :math:`L_1` regularization strength.  As shown
above, we can specify it in the ``run`` method using the ``hyperparams_prox``
argument The remaining arguments are passed to the objective function, here
``least_squares``.

Numerous proximal operators are available, see below.

Differentiation
~~~~~~~~~~~~~~~

In some applications, it is useful to differentiate the solution of the solver
with respect to some hyperparameters.  Continuing the previous example, we can
now differentiate the solution w.r.t. ``l1reg``::

  def solution(l1reg):
    pg = ProximalGradient(fun=least_squares, prox=prox_lasso, implicit_diff=True)
    return pg.run(w_init, hyperparams_prox=l1reg, data=(X, y)).params

  print(jax.jacobian(solution)(l1reg))

Under the hood, we use the implicit function theorem if ``implicit_diff=True``
and autodiff of unrolled iterations if ``implicit_diff=False``.  See the
:ref:`implicit differentiation <implicit_diff>` section for more details.

.. topic:: Examples

   * :ref:`sphx_glr_auto_examples_implicit_diff_lasso_implicit_diff.py`
   * :ref:`sphx_glr_auto_examples_implicit_diff_sparse_coding.py`

When using implicit differentiation, you can optionally specify a support
function ``support`` to give a hint to the linear solver called in ``root_vjp``
and only solve the linear system restricted to the support of the solution::

  from jaxopt.support import support_nonzero

  def solution(l1reg):
    pg = ProximalGradient(fun=least_squares, prox=prox_lasso,
                          support=support_nonzero, implicit_diff=True)
    return pg.run(w_init, hyperparams_prox=l1reg, data=(X, y)).params

  # Both the solution & the Jacobian have the same support
  print(solution(l1reg))
  print(jax.jacobian(solution)(l1reg))

See the :ref:`implicit differentiation <implicit_diff>` section for more details.

.. _block_coordinate_descent:

Block coordinate descent
------------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.BlockCoordinateDescent

Contrary to other solvers, :class:`jaxopt.BlockCoordinateDescent` only works with
:ref:`composite linear objective functions <composite_linear_functions>`.

Example::

  from jaxopt import objective
  from jaxopt import prox

  l1reg = 1.0
  w_init = jnp.zeros(n_features)
  bcd = BlockCoordinateDescent(fun=objective.least_squares, block_prox=prox.prox_lasso)
  lasso_sol = bcd.run(w_init, hyperparams_prox=l1reg, data=(X, y)).params

.. topic:: Examples

   * :ref:`sphx_glr_auto_examples_constrained_multiclass_linear_svm.py`
   * :ref:`sphx_glr_auto_examples_constrained_nmf.py`

Proximal operators
------------------

Proximal gradient and block coordinate descent do not access :math:`g(x, \lambda)`
directly but instead require its associated proximal operator. It is defined as:

.. math::

    \text{prox}_{g}(x', \lambda, \eta) :=
    \underset{x}{\text{argmin}} ~ \frac{1}{2} ||x' - x||^2 + \eta g(x, \lambda).

The following operators are available.

.. autosummary::
  :toctree: _autosummary

    jaxopt.prox.make_prox_from_projection
    jaxopt.prox.prox_none
    jaxopt.prox.prox_lasso
    jaxopt.prox.prox_non_negative_lasso
    jaxopt.prox.prox_elastic_net
    jaxopt.prox.prox_group_lasso
    jaxopt.prox.prox_ridge
    jaxopt.prox.prox_non_negative_ridge

.. _support_functions:

Support functions
-----------------

Support functions of the form :math:`S(x)` that returns 1 for all the
coordinates of :math:`x` in the support, and 0 otherwise:

.. math::

  S(x)_{j} := \begin{cases} 1 & \textrm{if $x_{j} \in S$} \\ 0 & \textrm{otherwise} \end{cases}

The following support functions are available.

.. autosummary::
  :toctree: _autosummary

    jaxopt.support.support_all
    jaxopt.support.support_nonzero
    jaxopt.support.support_group_nonzero
