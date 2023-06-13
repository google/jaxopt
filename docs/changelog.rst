Changelog
=========

Version 0.7
-----------

New features
~~~~~~~~~~~~

- Added :class:`jaxopt.LBFGSB`, by Emily Fertig.
- Added :func:`jaxopt.perturbations.make_perturbed_fun`, by Quentin Berthet.

Bug fixes and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Allow to pass a function as value_and_grad option, by Chansoo Lee.
- Fixed imaml tutorial (speed and correctness), by Zaccharie Ramzi.
- Misc improvements in resnet_flax example, by Fabian Pedregosa.
- Fixed prox to handle pytrees, by Vincent Roulet.
- Added control variate to make_perturbed_argmax, by Lawrence Stewart.
- Added inverse hessian approximation to the returned state, Aymeric Galan.
- Avoid closing over dynamic jax tracers in the bisection solver, by Roy Frostig.
- Follow pjit API changes, by Yash Katariya and Peter Hawkins.
- Added isotonic module to documentation, by Mathieu Blondel.

Contributors
~~~~~~~~~~~~

Aymeric Galan, Chansoo Lee, Emily Fertig, Fabian Pedregosa,
Lawrence Stewart, Mathieu Blondel, Peter Hawkins, Quentin Berthet,
Roy Frostig, Vincent Roulet, Yash Katariya, Zaccharie Ramzi.

Version 0.6
-----------

New features
~~~~~~~~~~~~

- Added new Hager-Zhang linesearch in LBFGS, by Srinivas Vasudevan (code review by Emily Fertig).
- Added perceptron and hinge losses, by Quentin Berthet.
- Added binary sparsemax loss, sparse_plus and sparse_sigmoid, by Vincent Roulet.
- Added isotonic regression, by Michael Sander.

Bug fixes and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added TPU support to notebooks, by Ayush Shridhar.
- Allowed users to restart from a previous optimizer state in LBFGS, by Zaccharie Ramzi.
- Added faster error computation in gradient descent algorithm, by Zaccharie Ramzi.
- Got rid of extra function call in BFGS and LBFGS, by Zaccharie Ramzi.
- Improved dtype consistency between input and output of update method, by Mathieu Blondel.
- Added perturbed optimizers notebook and narrative documentation, by Quentin Berthet and Fabian Pedregosa.
- Enabled auxiliary value returned by linesearch methods, by Zaccharie Ramzi.
- Added distributed examples to the website, by Fabian Pedregosa.
- Added Custom loop pjit example, by Felipe Llinares.
- Fixed wrong latex in maml.ipynb, by Fabian Pedregosa.
- Fixed bug in backtracking line search, by Srinivas Vasudevan (code review by Emily Fertig).
- Added pylintrc to top level directory, by Fabian Pedregosa.
- Corrected the condition function in LBFGS, by Zaccharie Ramzi.
- Added custom loop pmap example, by Felipe Llinares.
- Fixed pytree support in IterativeRefinement, by Louis Béthune.
- Fixed has_aux support in ArmijoSGD, by Louis Béthune.
- Documentation improvements, by Fabian Pedregosa and Mathieu Blondel.

Contributors
~~~~~~~~~~~~

Ayush Shridhar, Fabian Pedregosa, Felipe Llinares, Louis Bethune,
Mathieu Blondel, Michael Sander, Quentin Berthet, Srinivas Vasudevan, Vincent Roulet, Zaccharie Ramzi.

Version 0.5.5
-------------

New features
~~~~~~~~~~~~

- Added MAML example by Fabian Pedregosa based on initial code by Paul Vicol and Eric Jiang.
- Added the possibility to stop LBFGS after a line search failure, by Zaccharie Ramzi.
- Added gamma to LBFGS state, by Zaccharie Ramzi.
- Added :class:`jaxopt.BFGS`, by Mathieu Blondel.
- Added value_and_grad option to all gradient-based solvers, by Mathieu Blondel.
- Added Fenchel-Young loss, by Quentin Berthet.
- Added :func:`projection_sparse_simplex <jaxopt.projection.projection_sparse_simplex>`, by Tianlin Liu.

Bug fixes and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fixed missing args,kwargs in resnet example, by Louis Béthune.
- Corrected the implicit diff examples, by Zaccharie Ramzi.
- Small optimization in l2-regularized semi-dual OT, by Mathieu Blondel.
- Numerical stability improvements in :class:`jaxopt.LevenbergMarquardt`, by Amir Saadat.
- Dtype consistency in LBFGS, by Alex Botev.

Deprecations
~~~~~~~~~~~~

- ``jaxopt.QuadraticProgramming`` is now fully removed. Use
  :class:`jaxopt.CvxpyQP`, :class:`jaxopt.OSQP`, :class:`jaxopt.BoxOSQP` and
  :class:`jaxopt.EqualityConstrainedQP` instead.

Contributors
~~~~~~~~~~~~

Alex Botev, Amir Saadat, Fabian Pedregosa, Louis Béthune, Mathieu Blondel, Quentin Berthet, Tianlin Liu, Zaccharie Ramzi.

Version 0.5
-----------

New features
~~~~~~~~~~~~

- Added optimal transport related projections:
  :func:`projection_transport <jaxopt.projection.projection_transport>`,
  :func:`projection_birkhoff <jaxopt.projection.projection_birkhoff>`,
  :func:`kl_projection_transport <jaxopt.projection.kl_projection_transport>`,
  and
  :func:`kl_projection_birkhoff <jaxopt.projection.kl_projection_birkhoff>`,
  by Mathieu Blondel (semi-dual formulation) and Tianlin Liu (dual formulation).

Bug fixes and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fix LaTeX rendering issue in notebooks, by Amélie Héliou.
- Avoid gradient recompilations in zoom line search, by Mathieu Blondel.
- Fix unused Jacobian issue in :class:`jaxopt.ScipyRootFinding`, by Louis Béthune.
- Use zoom line search by default in :class:`jaxopt.LBFGS` and :class:`jaxopt.NonlinearCG`, by Mathieu Blondel.
- Pass tolerance argument to :class:`jaxopt.ScipyMinimize`, by pipme.
- Handle has_aux in :class:`jaxopt.LevenbergMarquardt`, by Keunhong Park.
- Add maxiter keyword argument in :class:`jaxopt.ScipyMinimize`, by Fabian Pedregosa.

Contributors
~~~~~~~~~~~~

Louis Béthune, Mathieu Blondel, Amélie Héliou, Keunhong Park, Fabian Pedregosa, pipme.

Version 0.4.3
-------------

New features
~~~~~~~~~~~~

- Added zoom line search in :class:`jaxopt.LBFGS`, by Mathieu Blondel. It can be enabled with the ``linesearch="zoom"`` option.

Bug fixes and enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added support for quadratic polynomial `fun` in :class:`jaxopt.BoxOSQP` and :class:`jaxopt.OSQP`, by Louis Béthune.
- Added a notebook for the dataset distillation example, by Amélie Héliou.
- Fixed wrong links and deprecation warnings in notebooks, by Fabian Pedregosa.
- Changed losses to avoid roundoff, by Jack Valmadre.
- Fixed init_params bug in multiclass_svm example, by Louis Béthune.


Contributors
~~~~~~~~~~~~

Louis Béthune, Mathieu Blondel, Amélie Héliou, Fabian Pedregosa, Jack Valmadre.


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

Felipe Llinares, Fabian Pedregosa, Ian Williamson, Louis Béthune, Mathieu Blondel, Roy Frostig.

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

Fabian Pedregosa, Felipe Llinares, Geoffrey Negiar, Louis Béthune, Mathieu
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
