Changelog
=========

Version 0.4.2
-------------

Bug fixes and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fix issue with positional arguments in :class:`jaxopt.LBFGS` and :class:`jaxopt.NonlinearCG`,
  by Mathieu Blondel.

Contributors
~~~~~~~~~~~~

Mathieu Blondel.

Version 0.4.1
-------------

Bug fixes and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Improvements in :class:`jaxopt.LBFGS`: fixed bug when using ``use_gamma=True``,
  added ``stepsize`` option, strengthened tests, by Mathieu Blondel.

- Fixed link in resnet notebook, by Fabian Pedregosa.

Contributors
~~~~~~~~~~~~

Fabian Pedregosa, Mathieu Blondel.


Version 0.4
-----------

New features
~~~~~~~~~~~~

- Added solver :class:`jaxopt.LevenbergMarquardt`, by Amir Saadat.
- Added solver :class:`jaxopt.BoxCDQP`, by Mathieu Blondel.
- Added :func:`projection_hypercube <jaxopt.projection.projection_hypercube>`, by Mathieu Blondel.

Bug fixes and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fixed :func:`solve_normal_cg <jaxopt.linear_solve.solve_normal_cg>`
  when the linear operator is "nonsquare" (does not map to a space of same dimension),
  by Mathieu Blondel.
- Fixed edge case in :class:`jaxopt.Bisection`, by Mathieu Blondel.
- Replaced deprecated tree_multimap with tree_map, by Fan Yang.
- Added support for leaf cond pytrees in :func:`tree_where <jaxopt.tree_util.tree_where>`, by Felipe Llinares.
- Added Python 3.10 support officially, by Jeppe Klitgaard.
- Replaced deprecated tree_multimap with tree_map, by Fan Yang.
- In scipy wrappers, converted pytree leaves to jax arrays to determine their shape/dtype, by Roy Frostig.
- Converted the "Resnet" and "Adversarial Training" examples to notebooks, by Fabian Pedregosa.

Contributors
~~~~~~~~~~~~

Amir Saadat, Fabian Pedregosa, Fan Yang, Felipe Llinares, Jeppe Klitgaard, Mathieu Blondel, Roy Frostig.

Version 0.3.1.
--------------

New features
~~~~~~~~~~~~

- Pjit-based example of data parallel training using Flax, by Felipe Llinares.

Bug fixes and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Support for GPU and state of the art adversarial training algorithm (PGD) on the robust_training.py example <https://github.com/google/jaxopt/pull/139>`_ by `Fabian Pedregosa <https://fa.bianp.net/>`_
- Update line search in LBFGS to use jit and unroll from LBFGS, by Ian Williamson.
- Support dynamic maximum iteration count in iterative solvers, by Roy Frostig.
- Fix tree_where for singleton pytrees, by Louis Béthune.
- Remove QuadraticProg in projections and set ``init_params=None`` by default in QP solvers, by Louis Béthune.
- Add missing 'value' attribute in LbfgsState, by Mathieu Blondel.

Contributors
~~~~~~~~~~~~

Felipe Llinares, Fabian Pedregosa, Ian Williamson, Louis Bétune, Mathieu Blondel, Roy Frostig.

Version 0.3
-----------

New features
~~~~~~~~~~~~

- :class:`jaxopt.LBFGS`
- :class:`jaxopt.BacktrackingLineSearch`
- :class:`jaxopt.GaussNewton`
- :class:`jaxopt.NonlinearCG`

Bug fixes and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Support implicit AD in higher-order differentiation
  <https://github.com/google/jaxopt/pull/143>`_.

Contributors
~~~~~~~~~~~~

Amir Saadat, Fabian Pedregosa, Geoffrey Négiar, Hyunsung Lee, Mathieu Blondel, Roy Frostig.

Version 0.2
-----------

New features
~~~~~~~~~~~~

- Quadratic programming solvers :class:`jaxopt.CvxpyQP`, :class:`jaxopt.OSQP`, :class:`jaxopt.BoxOSQP` and
  :class:`jaxopt.EqualityConstrainedQP`.
- :class:`jaxopt.IterativeRefinement`.

New examples
~~~~~~~~~~~~

- :ref:`sphx_glr_auto_examples_deep_learning_flax_resnet.py`

Bug fixes and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Prevent recompilation of loops in solver.run if executing without jit
  <https://github.com/google/jaxopt/pull/113>`_.
- `Prevents recomputation of gradient in OptaxSolver
  <https://github.com/google/jaxopt/pull/107>`_.
- `Make solver.update jittable and ensure output states are consistent
  <https://github.com/google/jaxopt/pull/106>`_.
- Allow ``Callable`` for the ``stepsize`` argument in
  :class:`jaxopt.ProximalGradient`, :class:`jaxopt.ProjectedGradient` and
  :class:`jaxopt.GradientDescent`.

Deprecations
~~~~~~~~~~~~

- :class:`jaxopt.QuadraticProgramming` is deprecated and will be removed in v0.4. Use
  :class:`jaxopt.CvxpyQP`, :class:`jaxopt.OSQP`, :class:`jaxopt.BoxOSQP` and
  :class:`jaxopt.EqualityConstrainedQP` instead.
- ``params, state = solver.init(...)`` is deprecated. Use ``state = solver.init_state(...)`` instead.

Contributors
~~~~~~~~~~~~

Fabian Pedregosa, Felipe Llinares, Geoffrey Negiar, Louis Bethune, Mathieu
Blondel, Vikas Sindhwani.

Version 0.1.1
-------------

New features
~~~~~~~~~~~~

- Added solver :class:`jaxopt.ArmijoSGD`
- Added example :ref:`sphx_glr_auto_examples_fixed_point_deep_equilibrium_model.py`
- Added example :ref:`sphx_glr_auto_examples_deep_learning_plot_sgd_solvers.py`

Bug fixes
~~~~~~~~~

- Allow non-jittable proximity operators in :class:`jaxopt.ProximalGradient`
- Raise an exception if a quadratic program is infeasible or unbounded

Contributors
~~~~~~~~~~~~

Fabian Pedregosa, Louis Bethune, Mathieu Blondel.

Version 0.1 (initial release)
-----------------------------

Classes
~~~~~~~

- :class:`jaxopt.AndersonAcceleration`
- :class:`jaxopt.AndersonWrapper`
- :class:`jaxopt.Bisection`
- :class:`jaxopt.BlockCoordinateDescent`
- :class:`jaxopt.FixedPointIteration`
- :class:`jaxopt.GradientDescent`
- :class:`jaxopt.MirrorDescent`
- :class:`jaxopt.OptaxSolver`
- :class:`jaxopt.PolyakSGD`
- :class:`jaxopt.ProjectedGradient`
- :class:`jaxopt.ProximalGradient`
- :class:`jaxopt.QuadraticProgramming`
- :class:`jaxopt.ScipyBoundedLeastSquares`
- :class:`jaxopt.ScipyBoundedMinimize`
- :class:`jaxopt.ScipyLeastSquares`
- :class:`jaxopt.ScipyMinimize`
- :class:`jaxopt.ScipyRootFinding`
- :ref:`Implicit differentiation <implicit_diff>`

Examples
~~~~~~~~

- :ref:`sphx_glr_auto_examples_constrained_binary_kernel_svm_with_intercept.py`
- :ref:`sphx_glr_auto_examples_deep_learning_flax_image_classif.py`
- :ref:`sphx_glr_auto_examples_deep_learning_haiku_image_classif.py`
- :ref:`sphx_glr_auto_examples_deep_learning_haiku_vae.py`
- :ref:`sphx_glr_auto_examples_implicit_diff_lasso_implicit_diff.py`
- :ref:`sphx_glr_auto_examples_constrained_multiclass_linear_svm.py`
- :ref:`sphx_glr_auto_examples_constrained_nmf.py`
- :ref:`sphx_glr_auto_examples_implicit_diff_plot_dataset_distillation.py`
- :ref:`sphx_glr_auto_examples_implicit_diff_ridge_reg_implicit_diff.py`
- :ref:`sphx_glr_auto_examples_implicit_diff_sparse_coding.py`
- :ref:`sphx_glr_auto_examples_deep_learning_robust_training.py`
- :ref:`sphx_glr_auto_examples_fixed_point_plot_anderson_accelerate_gd.py`
- :ref:`sphx_glr_auto_examples_fixed_point_plot_anderson_wrapper_cd.py`
- :ref:`sphx_glr_auto_examples_fixed_point_plot_picard_ode.py`

Contributors
~~~~~~~~~~~~

Fabian Pedregosa, Felipe Llinares, Louis Bethune, Marco Cuturi, Mathieu
Blondel, Peter Hawkins, Quentin Berthet, Robert Gower, Roy Frostig, Ta-Chu Kao
