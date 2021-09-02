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



.. topic:: Examples

   * :ref:`sphx_glr_auto_examples_plot_dataset_distillation.py`
   * :ref:`sphx_glr_auto_examples_lasso_implicit_diff.py`
   * :ref:`sphx_glr_auto_examples_ridge_reg_implicit_diff.py`
   * :ref:`sphx_glr_auto_examples_sparse_coding.py`



JAXopt solvers
--------------

All solvers in JAXopt support implicit differentiation out-of-the-box.
Most solvers have an ``implicit_diff=True|False`` option. When set to ``False``,
unrolling through the iterates is used instead of implicit differentiation.

..
  TODO: cross-reference lasso_implicit_diff

Custom solvers
--------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.implicit_diff.custom_root
    jaxopt.implicit_diff.custom_fixed_point

JAXopt also provides the ``custom_root`` and ``custom_fixed_point`` decorators,
for easily adding implicit differentiation on top of any existing solver.

..
  TODO: cross-reference ridge_reg_implicit_diff

JVPs and VJPs
-------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.implicit_diff.root_jvp
    jaxopt.implicit_diff.root_vjp
