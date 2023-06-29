# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax.test_util import check_grads
from jax.tree_util import tree_map
import numpy as onp

from sklearn import datasets
from sklearn import preprocessing
from sklearn import svm

from jaxopt import projection
from jaxopt._src.base import KKTSolution
from jaxopt._src.osqp import BoxOSQP
from jaxopt._src.osqp import OSQP
from jaxopt._src.osqp import extract_Qc_from_obj
from jaxopt._src.cvxpy_wrapper import CvxpyQP
from jaxopt._src import test_util


def get_random_osqp_problem(problem_size, eq_constraints, ineq_constraints):
  assert problem_size >= eq_constraints  # very likely to be infeasible
  onp.random.seed(problem_size + eq_constraints + ineq_constraints)
  Q = onp.random.randn(problem_size, problem_size)
  Q = Q.T.dot(Q)  # PSD matrix
  c = onp.random.randn(problem_size)
  A = onp.random.randn(ineq_constraints + eq_constraints, problem_size)
  l = onp.random.randn(ineq_constraints)
  u = l + jnp.abs(onp.random.randn(ineq_constraints))  # l < u
  b = onp.random.randn(eq_constraints)  # Ax = b
  l = jnp.concatenate([l, b])
  u = jnp.concatenate([u, b])
  return (Q, c), A, (l, u)


def _from_osqp_form_to_boxosqp_form(Q, c, A, l, u):
  """Return form used by CvxpyQP from BoxOSQP form (support only matrices)."""
  is_eq = l == u
  is_ineq_l = jnp.logical_and(l != u, l != -jnp.inf)
  is_ineq_u = jnp.logical_and(l != u, u != jnp.inf)
  if jnp.any(is_eq):
    A_eq, b = A[is_eq,:], l[is_eq]
  else:
    A_eq, b = jnp.zeros((1,len(c))), jnp.zeros((1,))
  if jnp.any(is_ineq_l) and jnp.any(is_ineq_u):
    G = jnp.concatenate([-A[is_ineq_l,:],A[is_ineq_u,:]])
    h = jnp.concatenate([-l[is_ineq_l],u[is_ineq_u]])
  elif jnp.any(is_ineq_l):
    G, h = -A[is_ineq_l,:], -l[is_ineq_l]
  elif jnp.any(is_ineq_u):
    G, h = A[is_ineq_u,:], u[is_ineq_u]
  else:
    return (Q, c), (A_eq, b), None
  return (Q, c), (A_eq, b), (G, h)


class BoxOSQPTest(test_util.JaxoptTestCase):

  def test_small_qp(self):
    # Setup a random QP min_x 0.5*x'*Q*x + q'*x s.t. Ax = z; l <= z <= u;
    eq_constraints, ineq_constraints = 2, 4
    onp.random.seed(42)
    problem_size = 16
    params_obj, params_eq, params_ineq = get_random_osqp_problem(problem_size, eq_constraints, ineq_constraints)
    tol = 1e-5
    osqp = BoxOSQP(tol=tol)
    params, state = osqp.run(None, params_obj, params_eq, params_ineq)
    self.assertLessEqual(state.error, tol)
    opt_error = osqp.l2_optimality_error(params, params_obj, params_eq, params_ineq)
    self.assertAllClose(opt_error, 0.0, atol=1e-4)

    def test_against_cvxpy(params_obj):
      (Q, c), Ab, Gh = _from_osqp_form_to_boxosqp_form(params_obj[0], params_obj[1],
                                                       params_eq, params_ineq[0], params_ineq[1])
      Q = 0.5 * (Q + Q.T)
      qp = CvxpyQP()
      hyperparams = dict(params_obj=(Q, c), params_eq=Ab, params_ineq=Gh)
      sol = qp.run(None, **hyperparams).params
      return sol.primal

    atol = 1e-4
    cvx_primal = test_against_cvxpy(params_obj)
    self.assertArraysAllClose(params.primal[0], cvx_primal, atol=atol)

    def osqp_run(params_obj):
      Q, c = params_obj
      Q = 0.5 * (Q + Q.T)
      sol = osqp.run(None, (Q, c), params_eq, params_ineq).params
      return sol.primal[0]

    jacosqp = jax.jacrev(osqp_run)(params_obj)
    jaccvxpy = jax.jacrev(test_against_cvxpy)(params_obj)
    self.assertArraysAllClose(jacosqp[0], jaccvxpy[0], atol=5e-2)
    self.assertArraysAllClose(jacosqp[1], jaccvxpy[1], atol=5e-2)

  @parameterized.product(derivative_with_respect_to=["eq", "ineq"])  # "none", "obj" are covered partially in other tests.
  def test_qp_eq_and_ineq(self, derivative_with_respect_to):
    Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
    c = jnp.array([1.0, 1.0])
    A_eq = jnp.array([[1.0, 1.0]])
    b = jnp.array([1.0])
    G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    h = jnp.array([0.0, 0.0])
    tol = 1e-5
    osqp = BoxOSQP(tol=tol, verbose=0)

    @jax.jit
    def osqp_run(Q, c, A_eq, G, b, h):
      l = jnp.concatenate([b, jnp.full(h.shape, -jnp.inf)])
      u = jnp.concatenate([b, jnp.full(h.shape, h)])
      Q = 0.5 * (Q + Q.T) # we need to ensure that P is symmetric even after directional perturbations
      A = jnp.concatenate([A_eq, G], axis=0)
      return osqp.run(None, (Q, c), A, (l, u))

    atol = 1e-2
    eps = 1e-4

    if "none" == derivative_with_respect_to:
      params, state = osqp_run(Q, c, A_eq, G, b, h)
      self.assertLessEqual(state.error, tol)
      assert state.status == BoxOSQP.SOLVED

      qp = CvxpyQP()
      hyperparams_qp = dict(params_obj=(Q, c), params_eq=(A_eq, b), params_ineq=(G, h))
      params_qp, state = qp.run(None, **hyperparams_qp)
      self.assertArraysAllClose(params.primal[0], params_qp.primal, atol=atol)

    def keep_ineq_only(params):
      b_idx = int(b.shape[0])
      mu, phi = params.dual_ineq
      return KKTSolution(params.primal, params.dual_eq, (mu[b_idx:], phi[b_idx:]))

    if "obj" == derivative_with_respect_to:
      solve_run_c = lambda c: keep_ineq_only(osqp_run(Q, c, A_eq, G, b, h).params)
      check_grads(solve_run_c, args=(c,), order=1, modes=['rev'], eps=eps, atol=atol)
      solve_run_Q = lambda Q: keep_ineq_only(osqp_run(Q, c, A_eq, G, b, h).params)
      check_grads(solve_run_Q, args=(Q,), order=1, modes=['rev'], eps=eps, atol=atol)

    if "eq" == derivative_with_respect_to:
      solve_run_A_eq = lambda A_eq: keep_ineq_only(osqp_run(Q, c, A_eq, G, b, h).params)
      check_grads(solve_run_A_eq, args=(A_eq,), order=1, modes=['rev'], eps=eps, atol=atol)
      solve_run_b = lambda b: keep_ineq_only(osqp_run(Q, c, A_eq, G, b, h).params)
      check_grads(solve_run_b, args=(b,), order=1, modes=['rev'], eps=eps, atol=atol)

    if "ineq" == derivative_with_respect_to:
      solve_run_G = lambda G: keep_ineq_only(osqp_run(Q, c, A_eq, G, b, h).params)
      check_grads(solve_run_G, args=(G,), order=1, modes=['rev'], eps=eps, atol=atol)
      solve_run_h = lambda h: keep_ineq_only(osqp_run(Q, c, A_eq, G, b, h).params)
      check_grads(solve_run_h, args=(h,), order=1, modes=['rev'], eps=eps, atol=atol)

  @parameterized.product(implicit_diff=[True, False])
  def test_projection_hyperplane(self, implicit_diff):
    x = jnp.array([1.0, 2.0])
    a = jnp.array([-0.5, 1.5])
    b = 0.3
    q = -x
    # Find ||y-x||^2 such that jnp.dot(y, a) = b.

    matvec_Q = lambda params_Q,u: u
    matvec_A = lambda params_A,u: jnp.dot(a, u).reshape(1)

    tol = 1e-4
    osqp = BoxOSQP(matvec_Q=matvec_Q, matvec_A=matvec_A, tol=tol, verbose=0)
    sol, state = osqp.run(None, (None, q), None, (b, b))

    assert state.status == BoxOSQP.SOLVED
    self.assertLessEqual(state.error, tol)
    atol = 1e-3
    opt_error = osqp.l2_optimality_error(sol, (None, q), None, (b, b))
    self.assertAllClose(opt_error, 0.0, atol=atol)
    expected = projection.projection_hyperplane(x, (a, b))
    self.assertArraysAllClose(sol.primal[0], expected, atol=atol)

  def test_projection_simplex(self):
    @jax.jit
    def _projection_simplex_qp(x, s=1.0):
      Q = jnp.eye(len(x))
      A_eq = jnp.array([jnp.ones_like(x)])
      b = jnp.array([s])
      G = -jnp.eye(len(x))
      h = jnp.zeros_like(x)
      A = jnp.concatenate([A_eq, G])
      l = jnp.concatenate([b, jnp.full(h.shape, -jnp.inf)])
      u = jnp.concatenate([b, h])
      hyperparams = dict(params_obj=(Q, -x), params_eq=A,
                         params_ineq=(l, u))

      osqp = BoxOSQP(tol=1e-5)
      return osqp.run(None, **hyperparams).params.primal[0]

    atol = 1e-3

    rng = onp.random.RandomState(0)
    x = jnp.array(rng.randn(10))
    p = projection.projection_simplex(x)
    p2 = _projection_simplex_qp(x)
    self.assertArraysAllClose(p, p2, atol=atol)

    J = jax.jacrev(projection.projection_simplex)(x)
    J2 = jax.jacrev(_projection_simplex_qp)(x)
    self.assertArraysAllClose(J, J2, atol=atol)

  def test_eq_constrained_qp_with_pytrees(self):
    rng = onp.random.RandomState(0)
    Q = rng.randn(7, 7)
    Q = onp.dot(Q, Q.T)
    A = rng.randn(4, 7)

    tmp = rng.randn(7)
    # Must have the same pytree structure as the output of matvec_P.
    c = {'foo':tmp[:3], 'bar':tmp[3:]}
    # Must have the same pytree structure as the output of matvec_A.
    b = [[rng.randn(1)] for _ in range(4)]

    def matvec_Q(Q, dic):
      x_ = jnp.concatenate([dic['foo'], dic['bar']])
      res = jnp.dot(Q, x_)
      return {'foo':res[:3], 'bar':res[3:]}

    def matvec_A(A, dic):
      x_ = jnp.concatenate([dic['foo'], dic['bar']])
      z = jnp.dot(A, x_)
      ineqs = jnp.split(z,z.shape[0])
      return [[ineq] for ineq in ineqs]

    tol = 1e-5
    atol = 1e-4

    # With pytrees directly.
    hyperparams = dict(params_obj=(Q, c), params_eq=A, params_ineq=(b, b))
    osqp = BoxOSQP(matvec_Q=matvec_Q, matvec_A=matvec_A, tol=tol)
    # sol.primal has the same pytree structure as the output of matvec_Q.
    # sol.dual_eq has the same pytree structure as the output of matvec_A.
    sol_pytree, state = osqp.run(None, **hyperparams)
    assert state.status == BoxOSQP.SOLVED
    self.assertAllClose(osqp.l2_optimality_error(sol_pytree, **hyperparams), 0.0, atol=atol)

    flat_x = lambda x: jnp.concatenate([x['foo'], x['bar']])
    flat_z = lambda z: jnp.concatenate([zi[0] for zi in z])

    # With flattened pytrees.
    c_flat = flat_x(c)
    b_flat = flat_z(b)
    hyperparams = dict(params_obj=(Q, c_flat), params_eq=A, params_ineq=(b_flat, b_flat))
    osqp = BoxOSQP(tol=tol)
    sol = osqp.run(None, **hyperparams).params
    self.assertAllClose(osqp.l2_optimality_error(sol, **hyperparams), 0.0, atol=atol)

    # Check that the solutions match.
    self.assertArraysAllClose(flat_x(sol_pytree.primal[0]), sol.primal[0], atol=atol)
    self.assertArraysAllClose(flat_z(sol_pytree.primal[1]), sol.primal[1], atol=atol)
    self.assertArraysAllClose(flat_z(sol_pytree.dual_eq), sol.dual_eq, atol=atol)

  def test_binary_kernel_svm(self):
    n_samples, n_features = 50, 8
    n_informative = (3*n_features) // 4
    # Prepare data.
    X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features,
                                        n_informative=n_informative, n_classes=2,
                                        random_state=0)
    X = preprocessing.Normalizer().fit_transform(X)
    y = y * 2 - 1.  # Transform labels from {0, 1} to {-1, 1}.
    lam = 10.0
    C = 1./ lam

    K = jnp.dot(X, X.T)

    # The dual objective is:
    # fun(beta) = 0.5 beta^T K beta - beta^T y
    # subject to
    # sum(beta) = 0
    # 0 <= beta_i <= C if y_i = 1
    # -C <= beta_i <= 0 if y_i = -1
    # where C = 1.0 / lam
    matvec_A = lambda _,b: (b, jnp.sum(b))
    l = -jax.nn.relu(-y * C), 0.
    u =  jax.nn.relu( y * C), 0.
    hyper_params = dict(params_obj=(K, -y), params_eq=None, params_ineq=(l, u))
    tol = 1e-5
    osqp = BoxOSQP(matvec_A=matvec_A, tol=tol)

    sol, state = osqp.run(None, **hyper_params)
    self.assertLessEqual(state.error, tol)
    atol = 1e-3
    opt_error = osqp.l2_optimality_error(sol, **hyper_params)
    self.assertAllClose(opt_error, 0.0, atol=atol)

    def binary_kernel_svm_skl(K, y):
      svc = svm.SVC(kernel="precomputed", C=C, tol=tol).fit(K, y)
      dual_coef = onp.zeros(K.shape[0])
      dual_coef[svc.support_] = svc.dual_coef_[0]
      return dual_coef

    beta_fit_skl = binary_kernel_svm_skl(K, y)
    # we solve the dual problem with BoxOSQP so the dual of svm.SVC
    # corresponds to the primal variables of BoxOSQP solution.
    self.assertAllClose(sol.primal[0], beta_fit_skl, atol=1e-2)

  def test_infeasible_polyhedron(self):
    # argmin_p \|p - x\|_2 = argmin_p <p,Ip> - 2<x,p> = argmin_p 0.5pIp - <x,p>
    # under p1 + p2 =  1
    #            p2 =  1
    #      -p1 + p2 = -1
    #      -p1     <=  0
    #      -p2     <=  0
    # This QP is primal/dual infeasible.
    A = jnp.array([[1.0, 1.0],[0.0, 1.0],[-1.0, 1.0]])
    b = jnp.array([1.0, 1.0, -1.0])
    G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    h = jnp.array([0.0, 0.0])

    l = b, jnp.full(h.shape, -jnp.inf)
    u = b, h

    def matvec_A(_, p):
      return jnp.dot(A, p), jnp.dot(G, p)

    x = jnp.zeros(2)
    I = jnp.eye(len(x))
    hyper_params = dict(params_obj=(I, -x), params_eq=None, params_ineq=(l, u))
    osqp = BoxOSQP(matvec_A=matvec_A, check_primal_dual_infeasability=True, tol=1e-5)
    sol, state = osqp.run(None, **hyper_params)
    assert state.status in [BoxOSQP.PRIMAL_INFEASIBLE, BoxOSQP.DUAL_INFEASIBLE]

  def test_infeasible_primal_only(self):
    # argmin   x1 + x2
    # under    x1 >= 6
    #          x2 >= 6
    #     x1 + x2 <= 11
    # This QP is primal infeasible.
    Q = jnp.zeros((2,2))
    c = jnp.array([1.,1.])

    def matvec_A(_, x):
      return x[0], x[1], x[0] + x[1]
    l = 6., 6., -jnp.inf
    u = jnp.inf, jnp.inf, 11.

    hyper_params = dict(params_obj=(Q, c), params_eq=None, params_ineq=(l, u))
    osqp = BoxOSQP(matvec_A=matvec_A, check_primal_dual_infeasability=True)
    sol, state = osqp.run(None, **hyper_params)
    assert state.status == BoxOSQP.PRIMAL_INFEASIBLE

  def test_unbounded_primal(self):
    # argmin   x1 -2x2 + x3
    # under  x1 + x2 >= 0
    #              x3 = 1
    # This (degenerated) QP is dual infeasible (unbounded primal).
    def matvec_A(_, x):
      return x[0] + x[1], x[2]
    l = 0., 1.
    u = jnp.inf, 1.

    def matvec_Q(_, x):
      return 0., 0., 0.

    hyper_params = dict(params_obj=(None, (-1., -2., 1.)), params_eq=None, params_ineq=(l, u))
    osqp = BoxOSQP(matvec_Q=matvec_Q, matvec_A=matvec_A,
                check_primal_dual_infeasability=True,
                tol=1e-3)
    init_params = osqp.init_params(init_x=(5., -3., 0.02), **hyper_params)
    sol, state = osqp.run(init_params, **hyper_params)
    assert state.status == BoxOSQP.DUAL_INFEASIBLE

  def test_jacobi_preconditioner(self):
    # Portfolio optimization
    # Problems of the form
    #     max_x mu^Tx - gamma x Cov x
    #     under sum_i x_i  = 1
    #                 x_i >= 0
    # with Cov = F@F.T + D the risk model, F @ F.T low rank, and D diagonal
    # mu the expected return
    onp.random.seed(0)
    assets, factors = 10, 3
    gamma = 1
    F = onp.random.randn(assets, factors)
    D = jnp.diag(onp.random.uniform(0., factors ** 0.5, assets))
    Cov = 2*gamma*(F @ F.T + D)   # minimize risk
    mu = onp.random.randn(assets)   # maximize gain

    A = jnp.concatenate([jnp.ones((1, assets)), jnp.eye(assets)], axis=0)
    l = jnp.concatenate([jnp.array([1.]), jnp.zeros(assets)])
    u = jnp.concatenate([jnp.array([1.]), jnp.full(assets, jnp.inf)])

    tol = 1e-5
    atol = 1e-3

    hyper_params = dict(params_obj=(Cov, mu), params_eq=A, params_ineq=(l, u))
    osqp = BoxOSQP(tol=tol, eq_qp_solve='cg+jacobi')
    sol, state = osqp.run(None, **hyper_params)
    self.assertLessEqual(state.error, tol)
    assert state.status == BoxOSQP.SOLVED
    opt_error = osqp.l2_optimality_error(sol, **hyper_params)
    self.assertAllClose(opt_error, 0.0, atol=atol)

    # Default preconditioner: Identity
    # Jacobi preconditioner J should be better.
    # We can check this heuristically with:
    #    || M@P@x - x ||_2 where preconditioner P approximates M^{-1}
    # Error for Default preconditioner:    M@x   - x
    # Error for Jacobi preconditioner:     M@J@x - x
    # The error of Jacobi is expected to be lower.
    sigma = 1e-6
    J = osqp._eq_qp_solve_impl
    x = sol.primal[0]
    for rho_bar in [1e-4, 0.001, 0.01, 0.1, 1., 10., 100., 1000, 1e4]:
      solver_state = J.init_state(x, Cov, A, sigma, rho_bar)
      M = Cov + sigma * jnp.eye(assets) + rho_bar * A.T @ A
      approx_x = M @ J._matvec_precond(solver_state, x)
      diff = jnp.sum((approx_x - x)**2)**0.5
      no_precond_approx = M @ x
      no_precond_diff = jnp.sum((no_precond_approx - x)**2)**0.5
      self.assertLessEqual(diff, no_precond_diff)

  def test_lasso(self):
    # Objective:
    #     min ||data @ x - targets||_2^2 + 2 * n * lam ||x||_1
    #
    # Converted in QP:
    #
    #     min y^Ty + 2*n*lam 1^T t
    #     under y = data @ x - targets
    #          -t <= x <= t
    #
    # With BoxOSQP formulation:
    #
    #     min y^Ty + 2*n*lam 1^T t
    #     under       targets = data @ x - y
    #           0         <= x + t <= infinity
    #           -infinity <= x - t <= 0
    #
    # Note that we use 2n corrective factor in objective to
    # optimize same objective as ProximalGradient.
    n = 10
    data, targets = datasets.make_regression(n_samples=n, n_features=3, random_state=0)
    lam = 10.0

    def matvec_Q(_, xyt):
      x, y, t = xyt
      return jnp.zeros_like(x), 2 * y, jnp.zeros_like(t)

    c = jnp.zeros(data.shape[1]), jnp.zeros(data.shape[0]), 2*n*lam * jnp.ones(data.shape[1])

    def matvec_A(data, xyt):
      x, y, t = xyt
      residuals = data @ x - y
      return residuals, x + t, x - t

    l = targets, jnp.zeros_like(c[0]), jnp.full(data.shape[1], -jnp.inf)
    u = targets, jnp.full(data.shape[1], jnp.inf), jnp.zeros_like(c[0])

    tol = 1e-4
    atol = 1e-2

    hyper_params = dict(params_obj=(None, c), params_eq=data, params_ineq=(l, u))
    osqp = BoxOSQP(matvec_Q=matvec_Q, matvec_A=matvec_A, tol=tol)
    sol, state = osqp.run(None, **hyper_params)
    self.assertLessEqual(state.error, tol)
    assert state.status == BoxOSQP.SOLVED
    opt_error = osqp.l2_optimality_error(sol, **hyper_params)
    self.assertAllClose(opt_error, 0.0, atol=atol)

    # Verify with Proximal Gradient

    def run_osqp(data):
      hyper_params = dict(params_obj=(None, c), params_eq=data, params_ineq=(l, u))
      sol, state = osqp.run(None, **hyper_params)
      return sol.primal[0][0]

    from jaxopt import objective
    from jaxopt import prox
    from jaxopt import ProximalGradient

    fun = objective.least_squares
    w_init = jnp.zeros(data.shape[1])
    pg = ProximalGradient(fun=fun, prox=prox.prox_lasso, maxiter=400, tol=tol,
                          acceleration=True)
    def run_prox(data):
      w_fit, info = pg.run(w_init, hyperparams_prox=lam, data=(data, targets))
      return w_fit

    w_fit = run_prox(data)
    self.assertArraysAllClose(sol.primal[0][0], w_fit, atol=atol)

    jac_osqp = jax.jacrev(run_osqp)(data)
    jac_prox = jax.jacrev(run_prox)(data)
    self.assertArraysAllClose(jac_osqp, jac_prox, atol=1e-2)

  def test_fun_api(self):
    hyper_params = dict(params_obj=None, params_eq=None, params_ineq=None)

    def fun(x, params_obj):
      del params_obj  # unused
      return 0.5*jnp.dot(x, x) + jnp.sum(x) + 17.0

    problem_size = 66
    x_init = jnp.arange(problem_size, dtype=jnp.float32)
    solver = BoxOSQP(fun=fun)
    (_, _, cste), c = extract_Qc_from_obj(x_init, hyper_params['params_obj'], solver.fun)

    self.assertAlmostEqual(cste, 17.0)
    self.assertArraysAllClose(c, jnp.ones(problem_size), atol=1e-3, rtol=1e-2)

  def test_convenience_api_random_problem(self):
    (Q, c), A , (l, u) = get_random_osqp_problem(3, 1, 1)

    params_obj = dict(foo=Q, bar=c)  # mimic arbitrary pytree.
    hyper_params = dict(params_obj=params_obj, params_eq=A, params_ineq=(l, u))
    cste = 42.
    def convex_quadratic(x, params_obj):
      Q, c = params_obj['foo'], params_obj['bar']
      return 0.5*jnp.dot(x, jnp.dot(Q, x)) + jnp.dot(c, x) + cste

    init_x = jnp.array([1., 2., 3])
    tol = 1e-4

    solver_with_fun = BoxOSQP(fun=convex_quadratic, tol=tol)
    (_, _, cste_approx), c_approx = extract_Qc_from_obj(init_x,
                                                        hyper_params['params_obj'],
                                                        solver_with_fun.fun)
    self.assertAlmostEqual(cste_approx, cste)
    self.assertArraysAllClose(c_approx, c, atol=1e-2, rtol=1e-2)

    try:
      # attempt to run without providing init_params, despite fun != None.
      init_params = solver_with_fun.init_params(init_x=None, **hyper_params)
    except ValueError as e:
      pass  # here, a failure is the expected behavior.
    else:
      self.fail("Expected ValueError when init_params is not provided (`fun` API).")

    init_params = solver_with_fun.init_params(init_x, **hyper_params)
    sol_with_fun, state_with_fun = solver_with_fun.run(init_params, **hyper_params)
    self.assertLessEqual(state_with_fun.error, tol)
    
    hyper_params_without_fun = dict(params_obj=(Q, c), params_eq=A, params_ineq=(l, u))
    solver_without_fun = BoxOSQP(tol=tol)
    init_params = solver_without_fun.init_params(init_x, **hyper_params_without_fun)
    sol_without_fun, state_without_fun = solver_without_fun.run(init_params, **hyper_params_without_fun)
    self.assertLessEqual(state_without_fun.error, tol)

    tree_map((lambda x,y: self.assertArraysAllClose(x, y, atol=1e-2)), sol_with_fun, sol_without_fun)

  def test_lu_factorization(self):
    problem_size = 200
    eq_constraints = 30
    ineq_constraints = 70
    params_obj, params_eq, params_ineq = get_random_osqp_problem(problem_size, eq_constraints, ineq_constraints)
    tol = 1e-4
    osqp = BoxOSQP(tol=tol, eq_qp_solve='lu', stepsize_updates_frequency=20)
    params, state = osqp.run(None, params_obj, params_eq, params_ineq)
    self.assertLessEqual(state.error, tol)
    opt_error = osqp.l2_optimality_error(params, params_obj, params_eq, params_ineq)
    self.assertAllClose(opt_error, 0.0, atol=1e-2)


class OSQPTest(test_util.JaxoptTestCase):

  def _check_derivative_A_b(self, solver, Q, c, A, b, params_ineq):
    @jax.jit
    def fun(Q, c, A, b):
      Q = 0.5 * (Q + Q.T)
      hyperparams = dict(params_obj=(Q, c),
                         params_eq=(A, b),
                         params_ineq=params_ineq)
      # reduce the primal variables to a scalar value for test purpose.
      return jnp.sum(solver.run(None, **hyperparams).params[0])

    # Derivative w.r.t. A.
    rng = onp.random.RandomState(0)
    V = rng.rand(*A.shape)
    V /= onp.sqrt(onp.sum(V ** 2))
    eps = 1e-3
    deriv_jax = jnp.vdot(V, jax.grad(fun, argnums=2)(Q, c, A, b))
    deriv_num = (fun(Q, c, A + eps * V, b) - fun(Q, c, A - eps * V, b)) / (2 * eps)
    self.assertAllClose(deriv_jax, deriv_num, rtol=1e-2)

    # Derivative w.r.t. b.
    v = rng.rand(*b.shape)
    v /= onp.sqrt(onp.sum(v ** 2))
    eps = 1e-3
    deriv_jax = jnp.vdot(v, jax.grad(fun, argnums=3)(Q, c, A, b))
    deriv_num = (fun(Q, c, A, b + eps * v) - fun(Q, c, A, b - eps * v)) / (2 * eps)
    self.assertAllClose(deriv_jax, deriv_num, rtol=1e-2)

  @parameterized.parameters((None, None), (jnp.dot, jnp.dot))
  def test_qp_eq_and_ineq(self, matvec_A, matvec_G):
    Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
    c = jnp.array([1.0, 1.0])
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([1.0])
    G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    h = jnp.array([0.0, 0.0])
    qp = OSQP(matvec_Q=None, matvec_A=matvec_A, matvec_G=matvec_G, tol=1e-5)
    hyperparams = dict(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h))
    sol = qp.run(None, **hyperparams).params
    self.assertAllClose(qp.l2_optimality_error(sol, **hyperparams), 0.0, atol=1e-4)
    self._check_derivative_A_b(qp, Q, c, A, b, (G, h))  # remark: slow.

  def test_projection_simplex(self):
    def _projection_simplex_qp(x, s=1.0):
      Q = jnp.eye(len(x))
      A = jnp.array([jnp.ones_like(x)])
      b = jnp.array([s])
      G = -jnp.eye(len(x))
      h = jnp.zeros_like(x)
      hyperparams = dict(params_obj=(Q, -x), params_eq=(A, b),
                         params_ineq=(G, h))

      qp = OSQP(tol=1e-5)
      # Returns the primal solution only.
      return qp.run(None, **hyperparams).params[0]

    rng = onp.random.RandomState(0)
    x = jnp.array(rng.randn(10))
    p = projection.projection_simplex(x)
    p2 = _projection_simplex_qp(x)
    self.assertArraysAllClose(p, p2, atol=1e-4)

    J = jax.jacrev(projection.projection_simplex)(x)
    J2 = jax.jacrev(_projection_simplex_qp)(x)
    self.assertArraysAllClose(J, J2, atol=1e-4)

  def test_NNLS(self):
    # Solve Non Negative Least Squares factorization.
    #
    #  min_W \|Y-UW\|_F^2
    #  s.t. W>=0
    n, m = 20, 10
    rank = 3
    onp.random.seed(654)
    U = jax.nn.relu(onp.random.randn(n, rank))
    W_0 = jax.nn.relu(onp.random.randn(m, rank))
    Y = U @ W_0.T

    def fun(W, params_obj):
      Y, U = params_obj
      return jnp.sum(jnp.square(Y - U @ W.T))

    def matvec_G(_, W):
      return -W

    zeros = jnp.zeros_like(W_0)
    hyper_params = dict(params_obj=(Y, U), params_eq=None, params_ineq=(None, zeros))

    init_W = jnp.zeros_like(W_0)
    solver = OSQP(fun=fun, matvec_G=matvec_G)
    init_params = solver.init_params(init_W, **hyper_params)
    sol, _ = solver.run(init_params=init_params, **hyper_params)
    W_sol = sol.primal

    # Check that the solution is close to the original.
    self.assertAllClose(W_0, W_sol, atol=1e-2, rtol=1e-2)

if __name__ == '__main__':
  jax.config.update("jax_enable_x64", False)
  absltest.main()
