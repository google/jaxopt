.. _implicit_diff:

Implicit differentiation
========================

Argmin differentiation
----------------------

Argmin differentiation is the task of differentiating a minimization problem's
solution with respect to its inputs. Namely, given

.. math::

    x^\star(\theta) := \underset{x}{\text{argmin}} f(x, \theta),

we would like to compute the Jacobian :math:`\partial x^\star(\theta)`.  This
is usually done either by implicit differentiation or by autodiff through an
algorithm's unrolled iterates.


JAXopt solvers
--------------

All solvers in JAXopt support implicit differentiation **out-of-the-box**.
Most solvers have an ``implicit_diff=True|False`` option. When set to ``False``,
autodiff of unrolled iterates is used instead of implicit differentiation.

Using the ridge regression example from the :ref:`unconstrained optimization
<unconstrained_optim>` section, we can write::

  def ridge_reg_objective(params, l2reg, X, y):
    residuals = jnp.dot(X, params) - y
    return jnp.mean(residuals ** 2) + 0.5 * l2reg * jnp.dot(params ** 2)

  def ridge_reg_solution(l2reg, X, y):
    gd = jaxopt.GradientDescent(fun=ridge_reg_objective, maxiter=500, implicit_diff=True)
    return gd.run(init_params, l2reg=l2reg, X=X, y=y).params

Now, ``ridge_reg_solution`` is differentiable just like any other JAX function.
Since ``ridge_reg_solution`` outputs a vector, we can compute its Jacobian::

  print(jax.jacobian(ridge_reg_solution, argnums=0)(l2reg, X, y)

where ``argnums=0`` specifies that we want to differentiate with respect to ``l2reg``.

We can also compose ``ridge_reg_solution`` with other functions::

  def validation_loss(l2reg):
    sol = ridge_reg_solution(l2reg, X_train, y_train)
    residuals = jnp.dot(X_val, params) - y_val
    return jnp.mean(residuals ** 2)

  print(jax.grad(validation_loss)(l2reg))

.. topic:: Examples

   * :ref:`sphx_glr_auto_examples_implicit_diff_plot_dataset_distillation.py`
   * :ref:`sphx_glr_auto_examples_implicit_diff_lasso_implicit_diff.py`
   * :ref:`sphx_glr_auto_examples_implicit_diff_sparse_coding.py`

Non-smooth functions
--------------------

When the function :math:`f(x, \theta)` is non-smooth (e.g., in Lasso), implicit
differentiation can still be applied to compute the Jacobian
:math:`\partial x^\star(\theta)`. However, the linear system to solve to obtain
the Jacobian (or, alternatively, the vector-Jacobian product) must be restricted
to the (generalized) *support* :math:`S` of the solution :math:`x^\star`. To
give a hint to the linear solver called in ``root_vjp``, you may specify a
support function ``support`` that will restrict the linear system to the support
of the solution.

The ``support`` function must return a pytree with the same structure and dtype
as the solution, where ``support(x)`` is equal to 1 for the coordinates :math:`j`
in the support (:math:`x_{j} \in S`), and 0 otherwise. The support function
depends on the non-smooth function being optimized; see :ref:`support functions
<support_functions>` for examples of support functions. Note that the
support function merely masks out coordinates outside of the support, making it
fully compatible with ``jit`` compilation.

Custom solvers
--------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.implicit_diff.custom_root
    jaxopt.implicit_diff.custom_fixed_point

JAXopt also provides the ``custom_root`` and ``custom_fixed_point`` decorators,
for easily adding implicit differentiation on top of any existing solver.

.. topic:: Examples

   * :ref:`sphx_glr_auto_examples_implicit_diff_ridge_reg_implicit_diff.py`

JVPs and VJPs
-------------

Finally, we also provide lower-level routines for computing the JVPs and VJPs
of roots of functions.

.. autosummary::
  :toctree: _autosummary

    jaxopt.implicit_diff.root_jvp
    jaxopt.implicit_diff.root_vjp

.. topic:: References:

 * `Efficient and Modular Implicit Differentiation
   <https://arxiv.org/abs/2105.15183>`_,
   Mathieu Blondel, Quentin Berthet, Marco Cuturi, Roy Frostig, Stephan Hoyer, Felipe Llinares-López, Fabian Pedregosa, Jean-Philippe Vert.
   ArXiv preprint.

 * `Implicit Differentiation for Fast Hyperparameter Selection in Non-Smooth Convex Learning
   <https://www.jmlr.org/papers/volume23/21-0486/21-0486.pdf>`_,
   Quentin Bertrand, Quentin Klopfenstein, Mathurin Massias, Mathieu Blondel, Samuel Vaiter, Alexandre Gramfort, Joseph Salmon.
   Journal of Machine Learning Research (JMLR).
