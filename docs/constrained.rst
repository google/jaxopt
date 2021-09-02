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

Projected gradient descent is gradient descent with an additional projection
onto the constraint set.

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

Projections always have two arguments: the input to be projected and the
parameters of the convex set.

Mirror descent
--------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.MirrorDescent


Quadratic programming
---------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.QuadraticProgramming

Equality-constrained QPs
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

    \min_{x} \frac{1}{2} x^\top Q x + c^\top x \textrm{ subject to } A x = b

Example::

  from jaxopt import QuadradicProgramming

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  A = jnp.array([[1.0, 1.0]])
  b = jnp.array([1.0])

  qp = QuadraticProgramming()
  sol = qp.run(params_obj=(Q, c), params_eq=(A, b)).params

  print(sol.primal)
  print(sol.dual_eq)


General QPs
~~~~~~~~~~~

.. math::

    \min_{x} \frac{1}{2} x^\top Q x + c^\top x \textrm{ subject to } A x = b, G x \le d

Example::

  from jaxopt import QuadradicProgramming

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  A = jnp.array([[1.0, 1.0]])
  b = jnp.array([1.0])
  G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
  h = jnp.array([0.0, 0.0])

  qp = QuadraticProgramming()
  sol = qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h)).params

  print(sol.primal)
  print(sol.dual_eq)
  print(sol.dual_ineq)

Scipy wrapper
-------------

For optimization with box constraints.

.. autosummary::
  :toctree: _autosummary

    jaxopt.ScipyBoundedMinimize
