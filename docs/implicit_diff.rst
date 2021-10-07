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
