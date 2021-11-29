Changelog
=========

Version 0.1.2 (development)
---------------------------

New features
~~~~~~~~~~~~

- Allow ``Callable`` for the ``stepsize`` argument in
  :class:`jaxopt.ProximalGradient`, :class:`jaxopt.ProjectedGradient` and
  :class:`jaxopt.GradientDescent`.
- Added example :ref:`sphx_glr_auto_examples_deep_learning_flax_resnet.py`

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
