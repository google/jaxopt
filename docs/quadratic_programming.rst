Quadratic programming
=====================

This section is concerned with minimizing quadratic functions subject
to equality and/or inequality constraints, also known as
`quadratic programming <https://en.wikipedia.org/wiki/Quadratic_programming>`_.

JAXopt supports several solvers for quadratic programming.
The solver specificities are summarized in the table below.
The best choice will depend on the usage.

.. list-table:: Quadratic programming solvers
   :widths: 45, 15, 20, 20, 15, 15, 15, 22, 15, 15
   :header-rows: 1

   * - Name
     - jit
     - pytree
     - matvec
     - quad. fun
     - precision
     - stability
     - speed
     - derivative
     - input format
   * - :class:`jaxopt.EqualityConstrainedQP`
     - yes
     - yes
     - yes
     - yes
     - ++
     - \+
     - +++
     - implicit
     - (Q, c), (A, b)
   * - :class:`jaxopt.CvxpyQP`
     - no
     - no
     - no
     - no
     - +++
     - +++
     - \+
     - implicit
     - (Q, c), (A, b), (G, h)
   * - :class:`jaxopt.OSQP`
     - yes
     - yes
     - yes
     - yes
     - \+
     - ++
     - ++
     - implicit
     - (Q, c), (A, b), (G, h)
   * - :class:`jaxopt.BoxOSQP`
     - yes
     - yes
     - yes
     - yes
     - \+
     - ++
     - ++
     - both
     - (Q, c), A, (l, u)
   * - :class:`jaxopt.BoxCDQP`
     - yes
     - no
     - no
     - no
     - ++
     - +++
     - ++
     - both
     - (Q, c), (l, u)

- *jit*: the algorithm can be used with jit or vmap, on GPU/TPU.
- *pytree*: the algorithm can be used with pytrees of matrices (see below).
- *matvec*: the QP parameters can be given as matvec instead of dense matrices (see below).
- *quad. fun*: the algorithm can be used with a quadratic function argument (see below).
- *precision*: accuracy expected when the solver succeeds to converge.
- *stability*: capacity to handle badly scaled problems and matrices with poor conditioning.
- *speed*: typical speed on big instances to reach its maximum accuracy.
- *derivative*: whether differentiation is supported only via implicit differentiation, or by both implicit differentiation and unrolling.
- *input format*: see subsections below.


This table is given as rule of thumb only; on some particular instances
some solvers may behave unexpectedly better (or worse!) than others.
In case of difficulties, we suggest to test different combinations of
algorithms, ``maxiter`` and ``tol`` values.

.. warning::

  These algorithms are guaranteed to converge on **convex problems** only.
  Hence, the matrix :math:`Q` *must* be positive semi-definite (PSD).

Equality-constrained QPs
------------------------

The problem takes the form:

.. math::

    \min_{x} \frac{1}{2} x^\top Q x + c^\top x \textrm{ subject to } A x = b

.. autosummary::
  :toctree: _autosummary

    jaxopt.EqualityConstrainedQP

This class is optimized for QPs with equality constraints only: it supports jit, pytrees and matvec.  
It is based on the KKT conditions of the problem.

Example::

  from jaxopt import EqualityConstrainedQP

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  A = jnp.array([[1.0, 1.0]])
  b = jnp.array([1.0])

  qp = EqualityConstrainedQP()
  sol = qp.run(params_obj=(Q, c), params_eq=(A, b)).params

  print(sol.primal)
  print(sol.dual_eq)

Ill-posed problems
~~~~~~~~~~~~~~~~~~

This solver is the fastest for well-posed problems, but can behave poorly on badly scaled matrices,
or with redundant constraints.

If the solver struggles to converge,
it is possible to enable
`iterative refinement <https://en.wikipedia.org/wiki/Iterative_refinement>`_.
This can be done by setting ``refine_regularization`` and ``refine_maxiter``::

  from jaxopt.eq_qp import EqualityConstrainedQP

  Q = 2 * jnp.array([[3000., 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  A = jnp.array([[1.0, 1.0]])
  b = jnp.array([1.0])

  qp = EqualityConstrainedQP(tol=1e-5, refine_regularization=3., refine_maxiter=50)
  sol = qp.run(params_obj=(Q, c), params_eq=(A, b)).params

  print(sol.primal)
  print(sol.dual_eq)
  print(qp.l2_optimality_error(sol, params_obj=(Q, c), params_eq=(A, b)))


General QPs
-----------

The problem takes the form:

.. math::

    \min_{x} \frac{1}{2} x^\top Q x + c^\top x \textrm{ subject to } A x = b, G x \le h

CvxpyQP
~~~~~~~

The wrapper over
`CVXPY <https://www.cvxpy.org>`_
is a solver that runs in ``float64`` precision.
However, it is not jittable, and does not support matvec and pytrees.

.. autosummary::
  :toctree: _autosummary

    jaxopt.CvxpyQP

Example::

  from jaxopt import CvxpyQP

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  A = jnp.array([[1.0, 1.0]])
  b = jnp.array([1.0])
  G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
  h = jnp.array([0.0, 0.0])

  qp = CvxpyWrapper()
  sol = qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h)).params

  print(sol.primal)
  print(sol.dual_eq)
  print(sol.dual_ineq)

It is also possible to specify only equality constraints or only inequality
constraints by setting ``params_eq`` or ``params_ineq`` to ``None``.

OSQP
~~~~

This solver is a pure JAX re-implementation of the OSQP algorithm.
It is jittable, supports pytrees and matvecs, but the precision is usually
lower than :class:`CvxpyQP` when run in float32 precision.
It is meant as a drop-in replacement for :class:`CvxpyQP`, but it
is a wrapper over :class:`BoxOSQP`.
Hence we recommend to use :class:`BoxOSQP` to avoid a costly problem transformation.

.. autosummary::
  :toctree: _autosummary

    jaxopt.OSQP

Example::

  from jaxopt import OSQP

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  A = jnp.array([[1.0, 1.0]])
  b = jnp.array([1.0])
  G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
  h = jnp.array([0.0, 0.0])

  qp = OSQP()
  sol = qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h)).params

  print(sol.primal)
  print(sol.dual_eq)
  print(sol.dual_ineq)

See :class:`jaxopt.BoxOSQP` for a full description of the parameters.

.. topic:: Example

   * :ref:`sphx_glr_auto_examples_constrained_multiclass_linear_svm.py`

Box-constrained QPs, with equality
----------------------------------

The problem takes the form:

.. math::

    \min_{x,z} \frac{1}{2} x^\top Q x + c^\top x \textrm{ subject to } A x = z, l \le z \le u

.. autosummary::
  :toctree: _autosummary

    jaxopt.BoxOSQP

:class:`jaxopt.BoxOSQP` uses the same underlying solver as :class:`jaxopt.OSQP`
but accepts problems in the above box-constrained format instead.  The bounds
``u`` (resp. ``l``) can be set to ``inf`` (resp. ``-inf``) if required.
Equality can be enforced with ``l = u``.

Example::

  from jaxopt import BoxOSQP

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  A = jnp.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
  l = jnp.array([1.0, -jnp.inf, -jnp.inf])
  u = jnp.array([1.0, 0.0, 0.0])

  qp = BoxOSQP()
  sol = qp.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params

  print(sol.primal)
  print(sol.dual_eq)
  print(sol.dual_ineq)

If required the algorithm can be sped up by setting
``check_primal_dual_infeasability`` to ``False``, and by setting
``eq_qp_preconditioner`` to ``"jacobi"`` (when possible).

.. note::

  The ``tol`` parameter controls the tolerance of the stopping criterion, which
  is based on the primal and dual residuals.  For over-constrained problems, or
  badly-scaled matrices, the residuals can be high, and it may be difficult to
  set ``tol`` appropriately.  In this case, it is better to tune ``maxiter``
  instead.

.. topic:: Example

  * :ref:`sphx_glr_auto_examples_constrained_binary_kernel_svm_with_intercept.py`

Box-constrained QPs, without equality
-------------------------------------

The problem takes the form:

.. math::

    \min_{x} \frac{1}{2} x^\top Q x + c^\top x \textrm{ subject to } l \le x \le u

.. autosummary::
  :toctree: _autosummary

    jaxopt.BoxCDQP

:class:`jaxopt.BoxCDQP` uses a coordinate descent solver. The solver returns only
the primal solution.

Example::

  from jaxopt import BoxCDQP

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, -1.0])
  l = jnp.array([0.0, 0.0])
  u = jnp.array([1.0, 1.0])

  qp = BoxCDQP()
  init = jnp.zeros(2)
  sol = qp.run(init, params_obj=(Q, c), params_ineq=(l, u)).params

  print(sol)

Unconstrained QPs
-----------------

For completeness, we also briefly describe how to solve unconstrained
quadratics of the form:

.. math::

    \min_{x} \frac{1}{2} x^\top Q x + c^\top x

The optimality condition rewrites :math:`\nabla \frac{1}{2} x^\top Q x + c^\top
x=Qx+c=0`.  Therefore, this is equivalent to solving the linear system
:math:`Qx=-c`.  Since the matrix :math:`Q` is assumed PSD, one of the best
algorithms is *conjugate gradient*.  In JAXopt, this can be done as follows::

  from jaxopt.linear_solve import solve_cg

  Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
  c = jnp.array([1.0, 1.0])
  matvec = lambda x: jnp.dot(Q, x)

  sol = solve_cg(matvec, b=-c)

  print(sol)

Pytree of matrices API
----------------------

Solvers :class:`EqualityConstrainedQP`, :class:`OSQP` and :class:`BoxOSQP` support
the pytree of matrices API. It means that the matrices `Q`, `A`, `G` can be provided
as block diagonal operator whose blocks are leaves of pytrees.
This corresponds to separable problems that can be solved in parallel (one for each leaf).

It offers several advantages:
  * This model of parallelism succeeds even if all the problems have different shapes,
    contrary to the `jax.vmap` API.
  * This formulation is more efficient than a single big matrix, especially when
    there are a lot of blocks, and when the blocks themselves are small.
  * The tolerance is globally defined and shared by all the problems,
    and the number of iterations is the same for all the problems. 

We illustrate below the parallel solving of two problems with different shapes::

  Q1 = jnp.array([[1.0, -0.5],
                  [-0.5, 1.0]])
  Q2 = jnp.array([[2.0]])
  Q = {'problem1': Q1, 'problem2': Q2}

  c1 = jnp.array([-0.4, 0.3])
  c2 = jnp.array([0.1])
  c = {'problem1': c1, 'problem2': c2}

  a1 = jnp.array([[-0.5, 1.5]])
  a2 = jnp.array([[10.0]])
  A = {'problem1': a1, 'problem2': a2}

  b1 = jnp.array([0.3])
  b2 = jnp.array([5.0])
  b = {'problem1': b1, 'problem2': b2}

  qp = EqualityConstrainedQP(tol=1e-3)
  hyperparams = dict(params_obj=(Q, c), params_eq=(A, b))
  # Solve the two problems in parallel with a single call.
  sol = qp.run(**hyperparams).params
  print(sol.primal['problem1'], sol.primal['problem2'])

Matvec API
----------

Solvers :class:`EqualityConstrainedQP`, :class:`OSQP` and :class:`BoxOSQP` support the matvec API.
It means that the user can provide a function ``matvec`` that computes the matrix-vector product,
either in the objective `x -> Qx` or in the constraints `x -> Ax`, `x -> Gx`.  
  
It offers several advantages:
  * the code is easier to read and closer to the mathematical formulation of the problem.
  * sparse matrix-vector products are available, which can be much faster than a dense one.
  * the derivatives w.r.t (params_obj, params_eq, params_ineq) may be easier to compute
    than materializing the full matrix.
  * it is faster than the quadratic function API.

This is the recommended API to use when the matrices are not block diagonal operators,
especially when there are other sparsity patterns involved, or in conjunction with
implicit differentiation::

  # Objective:
  #     min ||data @ x - targets||_2^2 + 2 * n * lam ||x||_1
  #
  # With BoxOSQP formulation:
  #
  #     min_{x, y, t} y^Ty + 2*n*lam 1^T t
  #     under       targets = data @ x - y
  #           0         <= x + t <= infinity
  #           -infinity <= x - t <= 0
  data, targets = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
  lam = 10.0

  def matvec_Q(params_Q, xyt):
    del params_Q  # unused
    x, y, t = xyt
    return jnp.zeros_like(x), 2 * y, jnp.zeros_like(t)

  c = jnp.zeros(data.shape[1]), jnp.zeros(data.shape[0]), 2*n*lam * jnp.ones(data.shape[1])

  def matvec_A(params_A, xyt):
    x, y, t = xyt
    residuals = params_A @ x - y
    return residuals, x + t, x - t

  l = targets, jnp.zeros_like(c[0]), jnp.full(data.shape[1], -jnp.inf)
  u = targets, jnp.full(data.shape[1], jnp.inf), jnp.zeros_like(c[0])

  hyper_params = dict(params_obj=(None, c), params_eq=data, params_ineq=(l, u))
  osqp = BoxOSQP(matvec_Q=matvec_Q, matvec_A=matvec_A, tol=1e-2)
  sol, state = osqp.run(None, **hyper_params)

Quadratic function API
----------------------

Solvers :class:`EqualityConstrainedQP`, :class:`OSQP` and :class:`BoxOSQP` support the quadratic function API.
It means that the whole objective function `x -> 1/2 x^T Q x + c^T x + K` can be provided as a function
``fun`` that computes the quadratic function. The function must be differentiable w.r.t `x`.

It offers several advantages:
  * the code is easier to read and closer to the mathematical formulation of the problem.
  * there is no need to provide the matrix `Q` and the vector `c` separately, nor to remove the constant term `K`.
  * the derivatives w.r.t (params_obj, params_eq, params_ineq) may be even easier to compute
    than materializing the full matrix.

Take care that this API also have drawbacks:
  * the function ``fun`` must be differentiable w.r.t `x` (with Jax's AD), even if you are not interested in the derivatives of your QP.
  * to extract `x -> Qx` and `c` from the function, we need to compute the Hessian-vector product and the gradient of ``fun``, which may be expensive.
  * for this API `init_params` must be provided to `run`, contrary to the other APIs.

We illustrate this API with Non Negative Least Squares (NNLS)::

  #  min_W \|Y-UW\|_F^2
  #  s.t. W>=0
  n, m, rank = 20, 10, 3
  onp.random.seed(654)
  U = jax.nn.relu(onp.random.randn(n, rank))
  W_0 = jax.nn.relu(onp.random.randn(rank, m))
  Y = U @ W_0

  def fun(W, params_obj):
    Y, U = params_obj
    # Write the objective as an implicit quadratic polynomial
    return jnp.sum(jnp.square(Y - U @ W))

  def matvec_G(params_G, W):
    del params_G  # unused
    return -W

  zeros = jnp.zeros_like(W_0)
  hyper_params = dict(params_obj=(Y, U), params_eq=None, params_ineq=(None, zeros))

  solver = OSQP(fun=fun, matvec_G=matvec_G)

  init_W = jnp.zeros_like(W_0)  # mandatory with `fun` API.
  init_params = solver.init_params(init_W, **hyper_params)
  W_sol = solver.run(init_params=init_params, **hyper_params).params.primal

This API is not recommended for large-scale problems or nested differentiations, use matvec API instead.

Implicit differentiation pitfalls
---------------------------------

When using implicit differentiation, the parameters w.r.t which we differentiate
must be passed to `params_obj`, `params_eq` or `params_ineq`. They should not be captured
from the global scope by `fun` or `matvec`. We illustrate below this common mistake::

  def wrong_solver(Q):  # don't do this!

    def matvec_Q(params_Q, x):
      del params_Q  # unused
      # error! Q is captured from the global scope.
      # it does not fail now, but it will fail later.
      return jnp.dot(Q, x)
    
    c = jnp.zeros(Q.shape[0])

    A = jnp.array([[1.0, 2.0]])
    b = jnp.array([1.0])

    eq_qp = EqualityConstrainedQP(matvec_Q=matvec_Q)
    sol = eq_qp.run(None, params_obj=(None, c), params_eq=(A, b)).params
    loss = jnp.sum(sol.primal)
    return loss

  Q = jnp.array([[1.0, 0.5], [0.5, 4.0]])
  _ = wrong_solver(Q)  # no error... but it will fail later.
  _ = jax.grad(wrong_solver)(Q)  # raise CustomVJPException

Also, notice that since the problems are convex, the optimum is independent of the
starting point `init_params`. Hence, derivatives w.r.t `init_params` are always
zero (mathematically).

The correct implementation is given below::

  def correct_solver(Q):

    def matvec_Q(params_Q, x):
      return jnp.dot(params_Q, x)
    
    c = jnp.zeros(Q.shape[0])

    A = jnp.array([[1.0, 2.0]])
    b = jnp.array([1.0])

    eq_qp = EqualityConstrainedQP(matvec_Q=matvec_Q)
    # Q is passed as a parameter, not captured from the global scope.
    sol = eq_qp.run(None, params_obj=(Q, c), params_eq=(A, b)).params
    loss = jnp.sum(sol.primal)
    return loss

  Q = jnp.array([[1.0, 0.5], [0.5, 4.0]])
  _ = correct_solver(Q)  # no error
  _ = jax.grad(correct_solver)(Q)  # no error
