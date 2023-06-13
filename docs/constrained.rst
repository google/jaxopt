.. _constrained_optim:

Constrained optimization
========================

This section is concerned with problems of the form

.. math::

    \min_{x} f(x, \theta) \textrm{ subject to } x \in \mathcal{C}(\upsilon),

where :math:`f(x, \theta)` is differentiable (almost everywhere), :math:`x` are
the parameters with respect to which the function is minimized, :math:`\theta`
are optional additional arguments, :math:`\mathcal{C}(\upsilon)` is a convex
set and :math:`\upsilon` are parameter the convex set may depend on.

Projected gradient
------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.ProjectedGradient

Instantiating and running the solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To solve constrained optimization problems, we can use projected gradient
descent, which is gradient descent with an additional projection onto the
constraint set. Constraints are specified by setting the ``projection``
argument. For instance, non-negativity constraints can be specified using
:func:`projection_non_negative <jaxopt.projection.projection_non_negative>`::

  from jaxopt import ProjectedGradient
  from jaxopt.projection import projection_non_negative

  pg = ProjectedGradient(fun=fun, projection=projection_non_negative)
  pg_sol = pg.run(w_init, data=(X, y)).params

Numerous projections are available, see below.

Specifying projection parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some projections have a hyperparameter that can be specified.  For
instance, the hyperparameter of :func:`projection_l2_ball
<jaxopt.projection.projection_l2_ball>` is the radius of the :math:`L_2` ball.
This can be passed using the ``hyperparams_proj`` argument of ``run``::

    from jaxopt.projection import projection_l2_ball

    radius = 1.0
    pg = ProjectedGradient(fun=fun, projection=projection_l2_ball)
    pg_sol = pg.run(w_init, hyperparams_proj=radius, data=(X, y)).params

.. topic:: Examples

  * :ref:`sphx_glr_auto_examples_constrained_binary_kernel_svm_with_intercept.py`

Differentiation
~~~~~~~~~~~~~~~

In some applications, it is useful to differentiate the solution of the solver
with respect to some hyperparameters.  Continuing the previous example, we can
now differentiate the solution w.r.t. ``radius``::

  def solution(radius):
    pg = ProjectedGradient(fun=fun, projection=projection_l2_ball, implicit_diff=True)
    return pg.run(w_init, hyperparams_proj=radius, data=(X, y)).params

  print(jax.jacobian(solution)(radius))

Under the hood, we use the implicit function theorem if ``implicit_diff=True``
and autodiff of unrolled iterations if ``implicit_diff=False``.  See the
:ref:`implicit differentiation <implicit_diff>` section for more details.

Projections
~~~~~~~~~~~

The Euclidean projection onto :math:`\mathcal{C}(\upsilon)` is:

.. math::

    \text{proj}_{\mathcal{C}}(x', \upsilon) :=
    \underset{x}{\text{argmin}} ~ ||x' - x||^2 \textrm{ subject to } x \in \mathcal{C}(\upsilon).

The following operators are available.

.. autosummary::
  :toctree: _autosummary

    jaxopt.projection.projection_non_negative
    jaxopt.projection.projection_box
    jaxopt.projection.projection_simplex
    jaxopt.projection.projection_sparse_simplex
    jaxopt.projection.projection_l1_sphere
    jaxopt.projection.projection_l1_ball
    jaxopt.projection.projection_l2_sphere
    jaxopt.projection.projection_l2_ball
    jaxopt.projection.projection_linf_ball
    jaxopt.projection.projection_hyperplane
    jaxopt.projection.projection_halfspace
    jaxopt.projection.projection_affine_set
    jaxopt.projection.projection_polyhedron
    jaxopt.projection.projection_box_section
    jaxopt.projection.projection_transport
    jaxopt.projection.projection_birkhoff

Projections always have two arguments: the input to be projected and the
parameters of the convex set.

Mirror descent
--------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.MirrorDescent

Kullback-Leibler projections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Kullback-Leibler projection onto :math:`\mathcal{C}(\upsilon)` is:

.. math::

    \text{proj}_{\mathcal{C}}(x', \upsilon) :=
    \underset{x}{\text{argmin}} ~ \text{KL}(x, \exp(x')) \textrm{ subject to } x \in \mathcal{C}(\upsilon).

The following operators are available.

.. autosummary::
  :toctree: _autosummary

    jaxopt.projection.kl_projection_transport
    jaxopt.projection.kl_projection_birkhoff

Box constraints
---------------

For optimization with box constraints, in addition to projected gradient
descent, we can use our SciPy wrapper.


.. autosummary::
  :toctree: _autosummary

    jaxopt.ScipyBoundedMinimize
    jaxopt.LBFGSB

This example shows how to apply non-negativity constraints, which can
be achieved by setting box constraints :math:`[0, \infty)`::

  from jaxopt import ScipyBoundedMinimize

  w_init = jnp.zeros(n_features)
  lbfgsb = ScipyBoundedMinimize(fun=fun, method="l-bfgs-b")
  lower_bounds = jnp.zeros_like(w_init)
  upper_bounds = jnp.ones_like(w_init) * jnp.inf
  bounds = (lower_bounds, upper_bounds)
  lbfgsb_sol = lbfgsb.run(w_init, bounds=bounds, data=(X, y)).params
