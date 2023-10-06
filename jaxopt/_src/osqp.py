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
"""GPU-friendly implementation of OSQP."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import jax
import jax.nn as nn
import jax.numpy as jnp
from jax.tree_util import tree_reduce

from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src.cond import cond
from jaxopt.tree_util import tree_add, tree_sub, tree_mul
from jaxopt.tree_util import tree_scalar_mul, tree_add_scalar_mul
from jaxopt.tree_util import tree_map, tree_vdot
from jaxopt.tree_util import tree_ones_like, tree_zeros_like, tree_where
from jaxopt.tree_util import tree_negative, tree_l2_norm, tree_inf_norm
from jaxopt._src.linear_operator import DenseLinearOperator, _make_linear_operator
import jaxopt.linear_solve as linear_solve


# Since jaxopt.projection itself depends on OSQP, we duplicate projection_box to avoid a circular dependency.
def _clip_safe(x, lower, upper):
  return jnp.clip(jnp.asarray(x), lower, upper)

def projection_box(x: Any, hyperparams: Tuple) -> Any:
  lower, upper = hyperparams
  return tree_map(_clip_safe, x, lower, upper)


def extract_Qc_from_obj(x: Any,
                        params_obj: Union[Tuple[Any, Any], Any],
                        fun: Optional[Callable]):
  """Returns (params_Q, c) parameters from params_obj.

  Args:
    x: pytree with same shape and same type as the input of fun; leaves can have any value.
    params_obj: parameters of objective function.
      Either a pair (params_Q, c) or an arbitrary pytree if fun is not None.
    fun: objective function.

  When `fun` is `None` it retrieves the relevant informations from the tuple `params_obj`,
  whereas when `fun` is not `None` it extracts it using AutoDiff.
  """
  if fun is None:
    params_Q, c = params_obj
    return params_Q, c
  zeros = tree_zeros_like(x)
  # f(x) := 0.5 * x^T Q x + c^T x + cste
  value_and_grad_fun = jax.value_and_grad(fun, argnums=0)
  # nabla_x f(x) = Q x + c
  # f(0)         = cste
  # Q is never extracted explicitly, only its action on vectors is needed
  cste, c = value_and_grad_fun(zeros, params_obj)
  return (params_obj, c, cste), c


def _make_osqp_optimality_fun(matvec_Q, matvec_A, fun):
  """Makes the optimality function for BoxOSQP.

  Args:
    matvec_Q: function that computes Qx.
    matvec_A: function that computes Ax.
    fun: function that computes the quadratic objective.

  Returns:
    optimality_fun(params, params_obj, params_eq, params_ineq) where
      params = (primal_var, eq_dual_var, ineq_dual_var)
      params_obj = (params_Q, c) or an arbitrary pytree if fun is not None
      params_eq = A
      params_ineq = (l, u)
  """

  if fun is None:
    def obj_fun(primal_var, params_obj):
      x, _ = primal_var
      params_Q, c = extract_Qc_from_obj(x, params_obj, fun)
      Q = matvec_Q(params_Q)
      # minimize 0.5 x^T Q x + c^T x
      qp_obj = tree_add_scalar_mul(tree_vdot(c, x), 0.5, tree_vdot(x, Q(x)))
      return qp_obj
  else:
    obj_fun = fun

  def eq_fun(primal_var, params_eq):
    # constraint Ax=z associated to y^T(Ax-z)=0
    x, z = primal_var
    A = matvec_A(params_eq)
    z_bar = A(x)
    return tree_sub(z_bar, z)

  def ineq_fun(primal_var, params_ineq):
    # constraints l<=z<=u associated to mu^T(z-u) + phi^T(l-z)
    _, z = primal_var
    l, u = params_ineq
    # if l=-inf (resp. u=+inf) then phi (resp. mu)
    # will be zero (never active at infinity)
    # so we may set the residual l-z (resp. z-u) to zero too.
    # since 0 * inf = 0 here.
    # but not in IEEE 754 standard where 0 * inf = nan.
    # Note: the derivative in +inf or -inf does not make sense anyway,
    # but those terms need to be computed since they are part of Lagrangian.
    u_inf = tree_map(lambda ui: ui != jnp.inf, u)
    l_inf = tree_map(lambda li: li != -jnp.inf, l)
    upper = tree_where(u_inf, tree_sub(z, u), 0)  # mu in dual
    lower = tree_where(l_inf, tree_sub(l, z), 0)  # phi in dual
    return upper, lower

  return idf.make_kkt_optimality_fun(obj_fun, eq_fun, ineq_fun)


class BoxOSQPState(NamedTuple):
  """Named tuple containing state information.

  Attributes:
    iter_num: iteration number.
    error: error used as stop criterion, deduced from residuals.
    status: integer, one of ``[BoxOSQP.UNSOLVED, BoxOSQP.SOLVED, BoxOSQP.PRIMAL_INFEASIBLE, BoxOSQP.DUAL_INFEASIBLE]``.
    primal_residuals: residuals of constraints of primal problem.
    dual_residuals: residuals of constraints of dual problem.
    rho_bar: current stepsize.
    solver_state: state of linear solver in the equality constrained QP
      that arises in ADMM iterations.
  """
  iter_num: int
  error: float
  status: int
  primal_residuals: Any
  dual_residuals: Any
  rho_bar: Any
  solver_state: Any


class OSQPLinearSolver:
  """Solve the linear system in OSQP.

    (Q + sigma I + rho_bar A^T A) x = b

  The system is solved repeatedly for different values of
  b or rho_bar during the execution of the algorithm.

  We leverage this property by carrying a state_linear_solver
  in BoxOSQPState that can be easily updated when rho_bar changes.
  """

  @abstractmethod
  def init_state(self, init_params, params_Q, params_A, sigma, rho_bar):
    pass

  @abstractmethod
  def update_stepsize(self, solver_state, rho_bar):
    pass

  @abstractmethod
  def run(self, b, osqp_state):
    pass


@dataclass(eq=False)
class OSQPIndirectSolver:
  """Indirect solver for Equality Constrained linear system in OSQP.

  Uses an indirect solver with warm start, with optional support for Jacobi preconditioning.
  Defaults to conjugate gradient descent.

  Attributes:
    solve: linear system solver to use (default: jax.scipy.sparse.linalg.cg).
    tol: tolerance of solver (default: 1e-7).
    maxiter: maximum number of iterations (default: None).
    jacobi_preconditioner: whether to use Jacobi preconditioning (default: False).
      Only available for pytree of matrices. Can significantly speed up the solver.
  """
  matvec_Q: Callable
  matvec_A: Callable
  solve: Callable = linear_solve.solve_cg
  tol: float = 1e-7
  maxiter: Optional[int] = None
  jacobi_preconditioner: bool = False

  def _init_state_precond(self, params_Q, params_A, sigma, rho_bar):
    if not self.jacobi_preconditioner:
      return None

    Q = self.matvec_Q(params_Q)
    A = self.matvec_A(params_A)
    if not isinstance(Q, DenseLinearOperator) or not isinstance(A, DenseLinearOperator):
      raise ValueError('Jacobi Preconditioning is only available for pytree of matrices.')

    precond_Q = Q.diag()
    precond_A = A.columns_l2_norms(squared=True)
    return precond_Q, precond_A

  def _matvec_precond(self, solver_state, x):
    if not self.jacobi_preconditioner:
      return x

    _, (_, _, sigma, rho_bar), (precond_Q, precond_A) = solver_state

    diag_precond = tree_add_scalar_mul(precond_Q, rho_bar, precond_A)
    diag_precond = tree_add(diag_precond, sigma)
    inv_diag = tree_map(lambda m_diag: 1/m_diag, diag_precond)
    return tree_mul(inv_diag, x)

  def init_state(self, init_params, params_Q, params_A, sigma, rho_bar):
    state_precond = self._init_state_precond(params_Q, params_A, sigma, rho_bar)
    return init_params, (params_Q, params_A, sigma, rho_bar), state_precond

  def update_stepsize(self, solver_state, rho_bar):
    prev_sol, (params_Q, params_A, sigma, _), state_precond = solver_state
    return prev_sol, (params_Q, params_A, sigma, rho_bar), state_precond

  def run(self, b, osqp_state):
    solver_state = osqp_state.solver_state
    prev_sol, (params_Q, params_A, sigma, rho_bar), state_precond = solver_state

    Q = self.matvec_Q(params_Q)
    A = self.matvec_A(params_A)

    primal_res_inf = tree_inf_norm(osqp_state.primal_residuals)
    dual_res_inf = tree_inf_norm(osqp_state.dual_residuals)
    lam = 0.15
    atol = lam * jnp.sqrt(primal_res_inf * dual_res_inf)
    # lam < 1 implies that atol is slower than geometric mean of primal_res_inf and dual_res_inf.

    def matvec_A(x_bar):
      Qx_sigmax = tree_add_scalar_mul(Q(x_bar), sigma, x_bar)
      ATAx = A.normal_matvec(x_bar)
      return tree_add_scalar_mul(Qx_sigmax, rho_bar, ATAx)

    sol = self.solve(matvec=matvec_A, b=b,
                     init=prev_sol,
                     M=lambda x: self._matvec_precond(solver_state, x),
                     maxiter=self.maxiter,
                     atol=atol,
                     tol=self.tol)
    solver_state = (sol, (params_Q, params_A, sigma, rho_bar), state_precond)
    return sol, solver_state


@dataclass(eq=False)
class OSQPLUSolver:
  """Solver based on LU factorization in OSQP.

  Updates LU factors when stepsize rho_bar changes."""

  def _lu_factor_dense(self, Q, A, sigma, rho_bar):
    dense = Q + sigma * jnp.eye(len(Q)) + rho_bar * A.T @ A
    return jax.scipy.linalg.lu_factor(dense)

  def _lu_factor_pytree(self, params_Q, params_A, sigma, rho_bar):
    lu_factor_dense = partial(self._lu_factor_dense, sigma=sigma, rho_bar=rho_bar)
    return tree_map(lu_factor_dense, params_Q, params_A)

  def init_state(self, init_params, params_Q, params_A, sigma, rho_bar):
    lu_factors = self._lu_factor_pytree(params_Q, params_A, sigma, rho_bar)
    return (params_Q, params_A, sigma), lu_factors

  def update_stepsize(self, solver_state, rho_bar):
    (params_Q, params_A, sigma), _ = solver_state
    lu_factors = self._lu_factor_pytree(params_Q, params_A, sigma, rho_bar)
    return (params_Q, params_A, sigma), lu_factors

  def run(self, b, osqp_state):
    _, lu_factors = osqp_state.solver_state

    # Switch order of b and lu_factors in call to lu_solve
    # to account for the fact that b is a prefix tree of lu_factors.
    def lu_solve(b, lu_factors):
      return jax.scipy.linalg.lu_solve(lu_factors, b, check_finite=False)

    sol = tree_map(lu_solve, b, lu_factors)

    return sol, osqp_state.solver_state


@dataclass(eq=False)
class BoxOSQP(base.IterativeSolver):
  """Operator Splitting Solver for Quadratic Programs.

  Jax implementation of the celebrated GPU-OSQP [1,3] based on ADMM.
  Suppports jit, vmap, matvecs, pytrees and fun.

  Refer to the doc of `init_state` method for the meaning of the parameters
  in `run` and `update` methods.

  It solves convex problems of the form

  .. math::

    \\begin{aligned}
      \\min_{x,z} \\quad & \\frac{1}{2}xQx + c^Tx\\\\
      \\textrm{s.t.} \\quad & Ax=z\\\\
        & l\\leq z\\leq u    \\\\
    \\end{aligned}

  Equality constraints are obtained by setting l = u.
  If the inequality is one-sided then ``jnp.inf can be used for u,
  and ``-jnp.inf`` for l.

  P must be a positive semidefinite (PSD) matrix.

  The Lagrangian is given by

  .. math::

    \\mathcal{L} = \\frac{1}{2}x^TQx + c^Tx + y^T(Ax-z) + \\mu^T (z-u) + \\phi^T (l-z)

  Primal    variables: :math:`x, z`

  Dual Eq   variables: :math:`y`

  Dual Ineq variables: :math:`\mu, \phi`


  ADMM computes :math:`y` at each iteration. :math:`\mu` and :math:`\phi` can be deduced from :math:`y`.

  Defaults values for hyper-parameters come from: https://github.com/osqp/osqp/blob/master/include/constants.h

  Attributes:
    matvec_Q: (optional) a Callable matvec_Q(params_Q, x).
      By default, matvec_Q(P, x) = tree_dot(P, x), where the pytree Q = params_Q matches x structure.
      The shape of primal variables may be inferred from params_obj = (matvec_Q, c).
    matvec_A: (optional) a Callable matvec_A(params_A, x).
      By default, matvec_A(A, x) = tree_dot(A, x), where tree pytree A = params_A matches x structure.
    fun: (optional) a function with signature fun(params, params_obj) that is promised
      to be a quadratic polynomial convex with respect to params, i.e fun can be written ::
        fun(x, params_obj) = 0.5*jnp.dot(x, jnp.dot(Q, x)) + jnp.dot(c, x) + cste
      with params_obj a pytree that contains the parameters of the objective function.
      (Q, c) do not need to be explicited in params_obj by the user: c will be inferred by Jaxopt,
        and the operator x -> Qx will be computed upon request.
      ``fun`` incompatible with the specification of ``matvec_Q``.
      Note that the shape of primal cannot be inferred from params_obj anymore,
      so the user should provide it in init_params.
      This API is provided for convenience, but note that since `fun` uses Jax's autodiff under the hood,
      it can be slower than `matvec_Q`, especially when used in conjunction with implicit differentiation.
    check_primal_dual_infeasability: if True populates the ``status`` field of ``state``
      with one of ``BoxOSQP.PRIMAL_INFEASIBLE``, ``BoxOSQP.DUAL_INFEASIBLE``.
      If False it improves speed but does not check feasability.
      If the problem is primal or dual infeasible, and jit=False, then a ValueError exception is raised.
      If "auto", it will be True if jit=False and False otherwise. (default: "auto")
    sigma: ridge regularization parameter in linear system.
    momentum: relaxation parameter (default: 1.6), must belong to the open interval (0,2).
      ``momentum=1`` => no relaxation.
      ``momentum<1`` => under-relaxation.
      ``momentum>1`` => over-relaxation.
      Boyd [2, p21] suggests chosing momentum in [1.5, 1.8].
    eq_qp_solve: 'cg', 'cg+jacobi' or 'lu' (default: 'cg').
      'cg' is conjugate gradient: an indirect solver that works with matvecs or pytree of matrices.
      'cg+jacobi' is conjugate gradient with Jacobi preconditioning: only works on pytree of matrices
        but can provide speedup.
      'lu' is LU factorization: a direct solver that only work on pytree of matrices.
    rho_start: initial learning rate  (default: 1e-1).
    rho_min: minimum learning rate  (default: 1e-6).
    rho_max: maximum learning rate  (default: 1e6).
    stepsize_updates_frequency: frequency of stepsize updates (default: 10).
      One every `stepsize_updates_frequency` updates computes a new stepsize.
    primal_infeasible_tol: relative tolerance for primal infeasability detection (default: 1e-3).
    dual_infeasible_tol: relative tolerance for dual infeasability detection (default: 1e-3).
    maxiter: maximum number of iterations (default: 4000).
    tol: absolute tolerance for stoping criterion (default: 1e-3).
    termination_check_frequency: frequency of termination check. (default: 5).
      One every `termination_check_frequency` the error is computed.
    verbose: If verbose=1 or True, print error at each iteration. 
      If verbose=2, also print stepsizes and primal/dual variables.
      If verbose=3, also print primal and dual residuals.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").

  References:

    [1] Stellato, B., Banjac, G., Goulart, P., Bemporad, A. and Boyd, S., 2020.
    OSQP: An operator splitting solver for quadratic programs.
    Mathematical Programming Computation, 12(4), pp.637-672.

    [2] Boyd, S., Parikh, N., Chu, E., Peleato, B. and Eckstein, J., 2010.
    Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers.
    Machine Learning, 3(1), pp.1-122.

    [3] Schubiger, M., Banjac, G. and Lygeros, J., 2020.
    GPU acceleration of ADMM for large-scale quadratic programming.
    Journal of Parallel and Distributed Computing, 144, pp.55-67.
  """
  matvec_Q: Optional[Callable] = None
  matvec_A: Optional[Callable] = None
  fun: Optional[Callable] = None
  check_primal_dual_infeasability: base.AutoOrBoolean = "auto"
  sigma: float = 1e-6
  momentum: float = 1.6
  eq_qp_solve: str = 'cg'
  rho_start: float = 0.1
  rho_min: float = 1e-6
  rho_max: float = 1e6
  stepsize_updates_frequency: int = 10
  primal_infeasible_tol: float = 1e-3
  dual_infeasible_tol: float = 1e-3
  maxiter: int = 4000
  tol: float = 1e-3
  termination_check_frequency: int = 5
  verbose: Union[bool, int] = False
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"


  # class attributes (ignored by @dataclass)
  UNSOLVED          = 0  # stopping criterion not reached yet.
  SOLVED            = 1  # feasible solution found with satisfying precision.
  DUAL_INFEASIBLE   = 2  # infeasible dual (infeasible primal or unbounded primal).
  PRIMAL_INFEASIBLE = 3  # infeasible primal.

  def init_state(self,
                 init_params: base.KKTSolution,
                 params_obj: Union[Tuple[Any, Any], Any],
                 params_eq: Any,
                 params_ineq: Tuple[Any, Any]):
    """Initializes the solver state.

    Args:
      init_params: initial primal and dual variables (KKTSolution).
      params_obj: parameters of the quadratic objective, can be:
        a tuple (Q, c) with Q a pytree of matrices,
        or a tuple (params_Q, c) if ``matvec_Q`` is provided,
        or an arbitrary pytree if ``fun`` is provided.
      params_eq: parameters of the equality constraints (see doc of run method).
      params_ineq: parameters of the inequality constraints (see doc of run method).

    Returns:
      A BoxOSQPState object.
    """
    x, z = init_params.primal
    y    = init_params.dual_eq
    Q_params, c = extract_Qc_from_obj(x, params_obj, self.fun)
    Q    = self.matvec_Q(Q_params)
    A    = self.matvec_A(params_eq)

    primal_residuals, dual_residuals = self._compute_residuals(Q, c, A, x, z, y)
    solver_state = self._eq_qp_solve_impl.init_state(x, Q_params, params_eq,
                                                     self.sigma, self.rho_start)

    return BoxOSQPState(iter_num=0,
                        error=jnp.inf,
                        status=BoxOSQP.UNSOLVED,
                        primal_residuals=primal_residuals,
                        dual_residuals=dual_residuals,
                        rho_bar=self.rho_start,
                        solver_state=solver_state)

  def init_params(self,
                  init_x: Any,
                  params_obj: Union[Tuple[Any, Any], Any],
                  params_eq: Any,
                  params_ineq: Tuple[Any, Any]):
    """Return default KKTSolution for initialization of the solver state.

    Args:
      init_x: initial primal variable.
      params_obj: parameters of the objective function (see doc of init_state method).
      params_eq: parameters of the equality constraints (see doc of init_state method).
      params_ineq: parameters of the inequality constraints (see doc of init_state method).

    Returns:
      init_params: default parameters for initialization.
    """
    if init_x is None:
      # assume params_obj = (params_Q, c)
      # generally incompatible with the use of ``fun``.
      if self.fun is not None:
        raise ValueError("init_x must be provided when fun is not None.")
      init_x = tree_zeros_like(params_obj[1])
    init_z = projection_box(self.matvec_A(params_eq)(init_x), params_ineq)
    init_y = tree_zeros_like(init_z)
    return base.KKTSolution((init_x, init_z), init_y, (init_y, init_y))

  def _get_full_KKT_solution(primal, y):
    """Returns all dual variables of the problem."""
    # Unfortunately BoxOSQP algorithm only returns y as dual variable,
    # mu and phi are missing, but can be recovered:
    #
    # We distinguish between l=u and l<u.
    # If l<u there are three cases:
    #   1. l < z < u: phi=0  mu=0 (and y=0)
    #   2. l = z < u: phi=-y mu=0 (and y<0)
    #   3. l < z = u: phi=0  mu=y (and y>0)
    #  this can be simplified with mu=relu(y) and phi=relu(-y)
    # If l=u then y=mu-phi, so we have one degree of liberty to chose mu and phi.
    # By symmetry with previous case we may chose mu=relu(y) and phi=relu(-y).
    is_pos = tree_map(lambda yi: yi >= 0, y)
    mu  = tree_where(is_pos, y, 0)  # derivative = 1 in y = 0
    phi = tree_map(lambda yi: jax.nn.relu(-yi), y)  # derivative = 0 in y = 0
    # y = mu - phi
    # d_y = d_mu - d_phi = 1 (everywhere; including in zero)
    return base.KKTSolution(primal=primal, dual_eq=y, dual_ineq=(mu, phi))

  def _update_stepsize(self, rho_bar, solver_state, primal_residuals, dual_residuals, Q, c, A, x, y):
    """Update stepsize based on the ratio between primal and dual residuals."""
    Ax, ATy     = A.matvec_and_rmatvec(x, y)
    primal_coef = tree_inf_norm(primal_residuals) / tree_inf_norm(Ax)
    max_inf     = jnp.maximum(tree_inf_norm(Q(x)), jnp.maximum(tree_inf_norm(ATy), tree_inf_norm(c)))
    dual_coef   = tree_inf_norm(dual_residuals) / max_inf
    eps_div     = jnp.finfo(dual_coef.dtype).eps
    coef        = jnp.sqrt(primal_coef / (dual_coef + eps_div))
    rho_bar     = jnp.clip(rho_bar * coef, self.rho_min, self.rho_max)
    solver_state = self._eq_qp_solve_impl.update_stepsize(solver_state, rho_bar)
    return rho_bar, solver_state

  def _compute_residuals(self, Q, c, A, x, z, y):
    """Compute residuals of constraints for primal and dual, as defined in paper."""
    Ax, ATy = A.matvec_and_rmatvec(x, y)
    primal_residuals = tree_sub(Ax, z)
    dual_residuals = tree_add(tree_add(Q(x), c), ATy)
    return primal_residuals, dual_residuals

  def _compute_error(self, primal_residuals, dual_residuals):
    """Return error based on primal/dual residuals."""
    primal_res_inf = tree_inf_norm(primal_residuals)
    dual_res_inf = tree_inf_norm(dual_residuals)
    criterion = jnp.maximum(primal_res_inf, dual_res_inf)
    status = jnp.where(criterion <= self.tol, BoxOSQP.SOLVED, BoxOSQP.UNSOLVED)
    return criterion, status

  def _check_dual_infeasability(self, error, status, delta_x, Q, c, Adx, l, u):
    criterion  = self.dual_infeasible_tol * tree_inf_norm(delta_x)

    certif_Q   = tree_inf_norm(Q(delta_x))
    certif_c   = tree_vdot(c, delta_x)

    unbounded_l = tree_map(lambda li: li == -jnp.inf, l)
    unbounded_u = tree_map(lambda ui: ui == jnp.inf, u)
    certif_l   = tree_map(lambda adxi,li: jnp.all(li <= adxi), Adx, tree_where(unbounded_l, -jnp.inf, -criterion))
    certif_u   = tree_map(lambda adxi,ui: jnp.all(adxi <= ui), Adx, tree_where(unbounded_u, jnp.inf, criterion))
    certif_A   = tree_reduce(jnp.logical_and, tree_map(jnp.logical_and, certif_l, certif_u))

    certif_dual_infeasible = jnp.logical_and(jnp.logical_and(certif_Q <= criterion, certif_c <= criterion), certif_A)

    if int(self.verbose) >= 2:
      jax.debug.print("certif_Q: {certif_Q} certif_c: {certif_c} certif_A: {certif_A} "
                      "criterion: {criterion}, Adx: {Adx}, certif_l: {certif_l}, certif_u: {certif_u}",
                      certif_Q=certif_Q, certif_c=certif_c, certif_A=certif_A, criterion=criterion,
                      Adx=Adx, certif_l=certif_l, certif_u=certif_u)

    # infeasible dual implies either infeasible primal, either unbounded primal.
    return jax.lax.cond(certif_dual_infeasible,
      lambda _: (0., BoxOSQP.DUAL_INFEASIBLE),  # dual unfeasible; exit the main loop with error = 0.
      lambda _: (error, status),
      operand=None)

  def _check_primal_infeasability(self, error, status, delta_y, ATdy, l, u):
    criterion = self.primal_infeasible_tol * tree_inf_norm(delta_y)
    certif_A  = tree_inf_norm(ATdy)
    bounded_l = tree_where(tree_map(lambda li: li == -jnp.inf, l), 0., l)  # replace inf bounds by zero
    bounded_u = tree_where(tree_map(lambda ui: ui == jnp.inf, u), 0., u)
    dy_plus   = tree_map(jax.nn.relu, delta_y)
    dy_minus  = tree_negative(tree_map(jax.nn.relu, tree_negative(delta_y)))
    certif_lu = tree_add(tree_vdot(bounded_l, dy_minus), tree_vdot(bounded_u, dy_plus))
    certif_primal_infeasible = jnp.logical_and(certif_A  <= criterion, certif_lu  <= criterion)

    if int(self.verbose) >= 2:
      jax.debug.print("certif_A: {certif_A}, certif_lu: {certif_lu}, criterion: {criterion}",
                      certif_A=certif_A, certif_lu=certif_lu, criterion=criterion)

    return jax.lax.cond(certif_primal_infeasible,
      lambda _: (0.,  # primal unfeasible; exit the main loop with error = 0.
                BoxOSQP.PRIMAL_INFEASIBLE),
      lambda _: (error, status),  # primal feasible or unbounded (depends of dual feasability).
      operand=None)

  def _check_infeasability(self, prev_sol, sol, error, status, Q, c, A, l, u):
    delta_x = tree_sub(sol.primal[0], prev_sol.primal[0])
    delta_y = tree_sub(sol.dual_eq, prev_sol.dual_eq)
    Adx, ATdy = A.matvec_and_rmatvec(delta_x, delta_y)

    error, status = self._check_dual_infeasability(error, status, delta_x, Q, c, Adx, l, u)
    error, status = self._check_primal_infeasability(error, status, delta_y, ATdy, l, u)

    return error, status

  def _check_termination_conditions(self, primal_residuals, dual_residuals,
                                    old_params, params, Q, c, A, l, u):
    error, status = self._compute_error(primal_residuals, dual_residuals)
    if self.check_primal_dual_infeasability:
      error, status = self._check_infeasability(old_params, params, error, status, Q, c, A, l, u)
    return error, status

  def _solve_linear_system(self, params, Q, c, A, rho_bar, state):
    """ Solve equality constrained QP in ADMM split."""
    # solve the "augmented" equality constrained QP:
    #
    #     minimize 0.5x_bar Q x_bar + c x_bar
    #     (1)        + (sigma/2) \|x_bar - x\|^2_2
    #     (2)        + (rho/2)   \|z_bar - z + rho^{-1} y\|^2_2
    #     under    A x_bar = z_bar; x_bar = x
    #
    #        (1) and (2) come from the augmented Lagrangian
    #
    # This problem is easy to solve by writing the KKT optimality conditions.
    # By construction the solution is unique without imposing strict convexity of objective nor
    # independance of the constraints.
    # The primal feasability conditions are used to eliminate z_bar from the system (which simplifies it).
    # Note that here we use rho_bar: we do not make use of per-constraint learning rate.
    x, z = params.primal
    y    = params.dual_eq  # dual variables for constraints z_bar = z;

    bxq = tree_sub(tree_scalar_mul(self.sigma, x), c)
    byz = tree_add_scalar_mul(y, -rho_bar, z)
    b = tree_sub(bxq, A.rmatvec(bxq, byz))

    x_bar, solver_state = self._eq_qp_solve_impl.run(b, state)

    return x_bar, solver_state

  def _admm_step(self, params, Q, c, A, box, rho_bar, state):
    """Performs one atomic step of the ADMM algorithm."""
    x, z = params.primal
    y    = params.dual_eq  # dual variables for constraints z_bar = z;
    # mu, phi = params.dual_ineq are unused

    # lines are numbered according to the pseudo-code in the paper OSQP: https://arxiv.org/pdf/1711.08013.pdf

    # line 3: optimization step for (x_bar, z_bar)
    # this equality constrained QP is solved by writing KKT conditions
    # which reduce to a well-posed linear system.
    x_bar, solver_state = self._solve_linear_system(params, Q, c, A, rho_bar, state)
    z_bar = A(x_bar)  # line 4

    # line 5: optimization step for x with relaxation parameter momentum (smooth updates)
    x_next = tree_add(x, tree_scalar_mul(self.momentum, tree_sub(x_bar, x)))

    # line 6: optimization step for z with relaxation parameter momentum (smooth updates)
    # by definition A x_bar = z_bar and l <= z <= u thanks to projection
    # the dual variable y corresponds to constraint z_bar = z
    z_momentum = tree_add(z, tree_scalar_mul(self.momentum, tree_sub(z_bar, z)))
    z_step = tree_scalar_mul(1/rho_bar, y)
    z_free = tree_add(z_momentum, z_step)
    z_next = projection_box(z_free, box)

    # line 7: gradient descent on dual variables, with relaxation
    y_step = tree_sub(z_momentum, z_next)
    y_next = tree_add(y, tree_scalar_mul(rho_bar, y_step))

    return (x_next, z_next), y_next, solver_state

  def update(self,
             params: base.KKTSolution,
             state: BoxOSQPState,
             params_obj: Union[Tuple[Any, Any], Any],
             params_eq: Any,
             params_ineq: Tuple[Any, Any]):
    """Perform BoxOSQP step."""
    # The original problem on variables (x,z) is split into TWO problems
    # with variables (x, z) and (x_bar, z_bar)
    #
    # (x_bar, z_bar) is NOT part of the state because it is recomputed at each step:
    #    (x_bar, z_bar) = argmin_{x_bar, z_bar} L(x_bar, z_bar, x, z, y)
    # with L the augmented Lagrangian
    # z_bar is always such that A x_bar = z_bar
    #
    # x = argmin_x L(x_bar, z_bar, x, z, y)
    # for equality constraint x = x_bar the dual variable is constant (=0) and can be eliminated
    #
    # z = argmin_z L(x_bar, z_bar, x, z, y)
    # for equality constraint z = z_bar the associated dual variable is y
    params_Q, c = extract_Qc_from_obj(params.primal[0], params_obj, self.fun)
    Q    = self.matvec_Q(params_Q)
    A    = self.matvec_A(params_eq)
    l, u = params_ineq

    # for active constraints (in particular equality constraints) high stepsize is better
    rho_bar = state.rho_bar
    if int(self.verbose) >= 2:
      jax.debug.print("rho_bar: {rho_bar}", rho_bar=rho_bar)

    (x, z), y, solver_state = self._admm_step(params, Q, c, A, (l, u), rho_bar, state)
    if int(self.verbose) >= 3:
      jax.debug.print("x: {x} z: {z} y: {y}", x=x, z=z, y=y)

    primal_residuals, dual_residuals = self._compute_residuals(Q, c, A, x, z, y)
    if int(self.verbose) >= 3:
      jax.debug.print("primal_residuals: {primal_residuals}, dual_residuals: {dual_residuals}",
                      primal_residuals=primal_residuals, dual_residuals=dual_residuals)

    # We need our own ifelse cond because automatic jitting of jax.lax.cond branches
    # could pose problems with non jittable matvecs, or prevent printing when verbose > 0.
    rho_bar, solver_state = cond(
        jnp.mod(state.iter_num, self.stepsize_updates_frequency) == 0,
        lambda _: self._update_stepsize(rho_bar, solver_state, primal_residuals, dual_residuals, Q, c, A, x, y),
        lambda _: (rho_bar, solver_state),
        None,
        jit=self.jit
    )

    sol = BoxOSQP._get_full_KKT_solution(primal=(x, z), y=y)

    # Same remark as above for ifelse cond.
    error, status = cond(
        jnp.mod(state.iter_num, self.termination_check_frequency) == 0,
        lambda _: self._check_termination_conditions(primal_residuals, dual_residuals,
                                                      params, sol, Q, c, A, l, u),
        lambda s: (state.error, s),
        state.status,
        jit=self.jit
    )

    if not self.jit:
      if status == BoxOSQP.PRIMAL_INFEASIBLE:
        raise ValueError(f"Primal infeasible.")
      if status == BoxOSQP.DUAL_INFEASIBLE:
        raise ValueError(f"Dual infeasible.")

    state = BoxOSQPState(iter_num=state.iter_num+1,
                         error=error,
                         status=status,
                         primal_residuals=primal_residuals,
                         dual_residuals=dual_residuals,
                         rho_bar=rho_bar,
                         solver_state=solver_state)
    return base.OptStep(params=sol, state=state)

  def run(self,
          init_params: Optional[Any] = None,
          params_obj: Tuple[Optional[Any], Any] = None,  # Q can be params_Q or None
          params_eq: Optional[Any] = None,  # A can be params_A or None
          params_ineq: Tuple[Any, Any] = None) -> base.OptStep:
    """Return primal/dual variables.

    Args:
      init_params: (optional) initial KKTSolution.
        Must be provided if ``fun`` is not None.
      params_obj: parameters of objective, can be:
        a tuple (Q, c) with Q a pytree of matrices,
        or a tuple (params_Q, c) if ``matvec_Q`` is provided,
        or an arbitrary pytree if ``fun`` is provided.
      params_eq: (optional) params_A.
      params_ineq: pair (l, u).
    """
    # Default to None for consistency of signatures
    # Should be set by the user anyway.
    assert params_obj is not None
    assert params_ineq is not None

    if init_params is None:
      init_params = self.init_params(None, params_obj, params_eq, params_ineq)

    return super().run(init_params, params_obj, params_eq, params_ineq)

  def l2_optimality_error(
      self,
      params: base.KKTSolution,
      params_obj: Tuple[Any, Any],
      params_eq: Any,
      params_ineq: Tuple[Any, Any]) -> base.OptStep:
    """Computes the L2 norm of the KKT residuals."""
    pytree = self.optimality_fun(params, params_obj, params_eq, params_ineq)
    return tree_l2_norm(pytree)

  def __post_init__(self):
    super().__post_init__()

    if self.fun is not None and self.matvec_Q is not None:
        raise ValueError(f"Specification of parameter 'fun' is incompatible with 'matvec_Q' in method __init__ of {type(self)}")

    if self.fun is not None:
      def matvec_Q(params_obj, x):
        params_Q, c, _ = params_obj
        # f(x) = 0.5 * x^T Q x + c^T x + cste
        # nabla_x f(x) = Q x + c
        # Qx = nabla_x f(x) - c
        def fun_minus_cx(xx):
          return self.fun(xx, params_Q) - jnp.sum(c*xx)
        Qx = jax.grad(fun_minus_cx)(x)
        return Qx
      self.matvec_Q = matvec_Q

    self.matvec_Q = _make_linear_operator(self.matvec_Q)
    self.matvec_A = _make_linear_operator(self.matvec_A)

    if self.eq_qp_solve.lower() == 'cg':
      self._eq_qp_solve_impl = OSQPIndirectSolver(self.matvec_Q, self.matvec_A,
                                                  tol=1e-7 * self.tol)
    elif self.eq_qp_solve.lower() == 'cg+jacobi':
      self._eq_qp_solve_impl = OSQPIndirectSolver(self.matvec_Q, self.matvec_A,
                                                  tol=1e-7 * self.tol,
                                                  jacobi_preconditioner=True)
    elif self.eq_qp_solve.lower() == 'lu':
      self._eq_qp_solve_impl = OSQPLUSolver()
    else:
      raise ValueError(f"Unknown solver '{self.eq_qp_solve}'.")

    if self.check_primal_dual_infeasability == "auto":
      self.check_primal_dual_infeasability = not self.jit

    self.optimality_fun = _make_osqp_optimality_fun(self.matvec_Q, self.matvec_A, self.fun)


class OSQP_to_BoxOSQP:
  """Converts a QP in OSQP / CvxpyQP form to a QP in BoxOSQP form."""

  @staticmethod
  def transform_matvec(matvec_Q, matvec_A, matvec_G):
    """Return matvec_Q, matvec_A of BoxOSQP from the matvec_Q, matvec_A, matvec_G of OSQP."""
    if matvec_A is None and matvec_G is None:
      matvec_A_box = None
    else:
      def matvec_A_box(params, x):
        params_A, params_G = params
        ret = []
        if matvec_A is not None:  # matvec_A.
          ret.append(matvec_A(params_A, x))
        elif params_A is not None:  # no matvec and pytree of matrices available.
          ret.append(DenseLinearOperator(params_A)(x))
        if matvec_G is not None:  # matvec_G.
          ret.append(matvec_G(params_G, x))
        elif params_G is not None:  # no matvec and pytree of matrices available.
          ret.append(DenseLinearOperator(params_G)(x))
        return ret  # list of length 1 or 2.

    return matvec_Q, matvec_A_box

  @staticmethod
  def _pytree_concat(pytrees, axis=0):
    """Concatenate leaves of a list of pytrees."""
    def concat(*leaves):
      return jnp.concatenate(leaves, axis=axis)
    return tree_map(concat, *pytrees)

  @staticmethod
  def transform(matvec_A_box: Optional[Callable],
                params: Optional[jnp.ndarray],
                params_obj: Tuple[Optional[Any], Any],
                params_eq: Optional[Tuple[Any,Any]],
                params_ineq: Optional[Tuple[Any,Any]]):
    """Transform parameters of run()"""
    # The huge volume of code is explained by the diversity of situations encountered:
    #
    # params can be None.
    # One of params_eq or params_ineq can be None.
    # One of A, G can be represented as a matvec, which forces matvec_A_box to be not None.
    # Note that params_A (resp. params_G) can be None if matvec_A (resp. matvec_G) is not None.
    # when matvec_A is None and matvec_G is None we MUST concatenate rows of constraints to ensure
    # that pre-conditioning is still possible.

    if params_eq is None and params_ineq is None:
      raise ValueError("At least one of params_eq or params_ineq must be not None." \
                       "When there are no constraints we suggest direct solving of the QP with a conjugate gradient solver.")

    eq_size, ineq_neg_size = None, None
    A, G = None, None
    l, u, y = [], [], []

    if params_eq is not None:
      A, b = params_eq
      l.append(b)
      u.append(b)
      eq_size = tree_map(lambda bi: bi.shape[0], b)
      if params is not None:
        y.append(params.dual_eq)

    if params_ineq is not None:
      G, h = params_ineq
      l.append(tree_scalar_mul(-jnp.inf, tree_ones_like(h)))
      u.append(h)
      ineq_neg_size = tree_map(lambda hi: -hi.shape[0], h)
      if params is not None:
        y.append(params.dual_ineq)

    A_box = [tree_map(jnp.asarray, A), tree_map(jnp.asarray, G)]
    if matvec_A_box is None:
      # no matvec: construct a pytree of matrices containing all constraints: A_box = [A; G].
      if None in A_box:
        A_box.remove(None)
      A_box = OSQP_to_BoxOSQP._pytree_concat(A_box)
      l = OSQP_to_BoxOSQP._pytree_concat(l)
      u = OSQP_to_BoxOSQP._pytree_concat(u)

    if params is not None:
      x = params.primal
      z = _make_linear_operator(matvec_A_box)(A_box)(x)
      if matvec_A_box is None:
        y = OSQP_to_BoxOSQP._pytree_concat(y)
      params = BoxOSQP._get_full_KKT_solution((x, z), y)

    hyper_params = dict(params_obj=params_obj, params_eq=A_box, params_ineq=(l, u))
    return params, hyper_params, (eq_size, ineq_neg_size)

  @staticmethod
  def _pytree_split(pytree, slice_sizes):
    """Extract slices of size slice_sizes in each leaf of pytree."""
    if slice_sizes is None:
      return None
    _signed_slice = lambda leaf,slice: (leaf[:slice] if slice > 0 else leaf[slice:])
    return tree_map(_signed_slice, pytree, slice_sizes)

  @staticmethod
  def inverse_transform(matvec_A_box, eq_ineq_size, kkt_solution):
    """Inverse transform the KKT solution returned by run()"""
    box_primal, box_dual_eq, (box_mu, _) = kkt_solution
    x, _ = box_primal

    eq_size, ineq_neg_size = eq_ineq_size
    if matvec_A_box is not None:
      box_dual_eq = box_dual_eq[0]  # if Ax=b is defined, it is here.
      box_mu = box_mu[-1]  # if Gx <= h is defined, it is here.
    # else: dual_variable is a pytree of (concatenated) tensors

    # The sign of ineq_neg_size removes the need of detecting if it is a concatenation or singleton.
    dual_eq = OSQP_to_BoxOSQP._pytree_split(box_dual_eq, eq_size)
    dual_ineq = OSQP_to_BoxOSQP._pytree_split(box_mu, ineq_neg_size)

    return base.KKTSolution(x, dual_eq, dual_ineq)


class OSQPState(NamedTuple):
  iter_num: int
  error: float
  status: int


class OSQP(base.Solver):
  """OSQP solver for general quadratic programming.

  Meant as drop-in replacement for CvxpyQP.
  Depending on the format of your problem, BoxOSQP API may be more appropriate.
  Support for matvec and pytrees. Supports jit and vmap.

  CvxpyQP is more precise and should be preferred on CPU.
  OSQP can be quicker than CvxpyQP when GPU/TPU are available.

  The objective function is::

    0.5 * x^T Q x + c^T x subject to Gx <= h, Ax = b.

  The attributes must be given as keyword arguments.
  Hyper-parameters defaults to the same values as in BoxQP.

  Attributes:
    matvec_Q: (optional) a Callable matvec_Q(params_Q, x).
      By default, matvec_Q(P, x) = tree_dot(P, x), where the pytree Q = params_Q matches x structure.
      ``matvec_Q`` incompatible with the specification of ``fun``.
      The shape of primal variables may be inferred from params_obj = (matvec_Q, c).
    matvec_A: (optional) a Callable matvec_A(params_A, x).
      By default, matvec_A(A, x) = tree_dot(A, x), where tree pytree A = params_A matches x structure.
    matvec_G: (optional) a Callable matvec_G(params_G, x).
      By default, matvec_G(G, x) = tree_dot(G, x), where tree pytree G = params_G matches x structure.
    fun: (optional) a function with signature fun(params, params_obj) that is promised
      to be quadratic polynomial convex with respect to params, i.e fun can be written ::
        fun(x, params_obj) = 0.5*jnp.dot(x, jnp.dot(Q, x)) + jnp.dot(c, x) + cste
      with params_obj a pytree that contains the parameters of the objective function.
      (Q, c) do not need to be explicited in params_obj by the user: c will be inferred by Jaxopt,
        and the operator x -> Qx will be computed upon request.
      ``fun`` incompatible with the specification of ``matvec_Q``.
      Note that the shape of primal cannot be inferred from params_obj anymore,
      so the user should provide it in init_params.
    check_primal_dual_infeasability: if True populates the ``status`` field of ``state``
      with one of ``BoxOSQP.PRIMAL_INFEASIBLE``, ``BoxOSQP.DUAL_INFEASIBLE``. (default: True).
      If False it improves speed but does not check feasability.
      If jit=False, and if the problem is primal or dual infeasible, then a ValueError exception is raised.
    sigma: ridge regularization parameter in linear system.
    momentum: relaxation parameter (default: 1.6), must belong to the open interval (0,2).
      momentum=1 => no relaxation.
      momentum<1 => under-relaxation.
      momentum>1 => over-relaxation.
      Boyd [2, p21] suggests chosing momentum in [1.5, 1.8].
    eq_qp_solve: 'cg', 'cg+jacobi' or 'lu' (default: 'cg').
      'cg' is conjugate gradient: an indirect solver that works with matvecs or pytree of matrices.
      'cg+jacobi' is conjugate gradient with Jacobi preconditioning: only works on pytree of matrices
        but can provide speedup.
      'lu' is LU factorization: a direct solver that only work on pytree of matrices.
    rho_start: initial learning rate. (default: 1e-1)
    rho_min: minimum learning rate. (default: 1e-6)
    rho_max: maximum learning rate. (default: 1e6)
    stepsize_updates_frequency: frequency of stepsize updates. (default: 10).
      One every `stepsize_updates_frequency` updates computes a new stepsize.
    primal_infeasible_tol: relative tolerance for primal infeasability detection. (default: 1e-4)
    dual_infeasible_tol: relative tolerance for dual infeasability detection. (default: 1e-4)
    maxiter: maximum number of iterations.  (default: 4000)
    tol: absolute tolerance for stoping criterion (default: 1e-3).
    termination_check_frequency: frequency of termination check. (default: 5).
      One every `termination_check_frequency` the error is computed.
    implicit_diff_solve: the linear system solver to use.
  """
  matvec_A_box: Optional[Callable] = None

  def __init__(self, *,
    matvec_Q: Optional[Callable] = None,
    matvec_A: Optional[Callable] = None,
    matvec_G: Optional[Callable] = None,
    fun: Optional[Callable] = None,
    **kwargs):
    if fun is not None and matvec_Q is not None:
      raise ValueError(f"Specification of parameter 'fun' is incompatible with 'matvec_Q' in method __init__ of {type(self)}")

    matvec_Q, matvec_A_box = OSQP_to_BoxOSQP.transform_matvec(matvec_Q, matvec_A, matvec_G)
    self.matvec_A_box = matvec_A_box

    # meant as drop-in replacement for CvxpyQP for which no unrolling is available.
    self._box_osqp = BoxOSQP(matvec_Q=matvec_Q,
                             matvec_A=matvec_A_box,
                             fun=fun,
                             implicit_diff=True,
                             **kwargs)

  def init_params(self, init_x, params_obj, params_eq, params_ineq):
    """Return default params for initialization.

    Args:
      init_x: initial primal solution.
      params_obj: see the doc of `run` method.
      params_eq: see the doc of `run` method.
      params_ineq: see the doc of `run` method.

    Returns:
      init_params: a pytree KKTSolution of parameters for BoxOSQP.
    """
    _, hyper_params, eq_ineq_size = OSQP_to_BoxOSQP.transform(self.matvec_A_box,
                                                              None, params_obj,
                                                              params_eq, params_ineq)
    init_params_box_osqp = self._box_osqp.init_params(init_x, **hyper_params)
    return OSQP_to_BoxOSQP.inverse_transform(self.matvec_A_box, eq_ineq_size, init_params_box_osqp)

  def run(self,
          init_params: Any = None,
          params_obj: Union[Tuple[Any, Any], Any] = None,
          params_eq: Optional[Tuple[Any,Any]] = None,
          params_ineq: Optional[Tuple[Any,Any]] = None) -> base.OptStep:
    """Runs the quadratic programming solver in BoxOSQP.

    The returned params contains both the primal and dual solutions.

    Args:
      init_params: init_params: (optional) initial KKTSolution for warm-start.
        Must be provided if ``fun`` is not None.
      params_obj: parameters of the quadratic objective, can be:
        a tuple (Q, c) with Q a pytree of matrices,
        or a tuple (params_Q, c) if ``matvec_Q`` is provided,
        or an arbitrary pytree if ``fun`` is provided.
      params_eq: (A, b) or None if no equality constraints.
      params_ineq: (G, h) or None if no inequality constraints.
    Returns:
      (params, state), ``params = (primal_var, dual_var_eq, dual_var_ineq)``
    """
    assert params_obj is not None
    if init_params is None:
      init_params = self.init_params(None, params_obj, params_eq, params_ineq)
    init_params, hyper_params, eq_ineq_size = OSQP_to_BoxOSQP.transform(self.matvec_A_box,
                                                                        init_params, params_obj,
                                                                        params_eq, params_ineq)
    sol, box_osqp_state = self._box_osqp.run(init_params, **hyper_params)
    sol = OSQP_to_BoxOSQP.inverse_transform(self.matvec_A_box, eq_ineq_size, sol)
    state = OSQPState(
      iter_num=box_osqp_state.iter_num,
      error=box_osqp_state.error,
      status=box_osqp_state.status
    )
    return base.OptStep(params=sol, state=state)

  def l2_optimality_error(
      self,
      params: jnp.ndarray,
      params_obj: Union[base.ArrayPair, Any],
      params_eq: Optional[base.ArrayPair],
      params_ineq: Optional[base.ArrayPair],
  ):
    """Computes the L2 norm of the KKT residuals.

    Note that this function is exposed for consistency of the API, but the differentiation is actually
    performed in the BoxOSQP class."""
    params, hyper_params, _ = OSQP_to_BoxOSQP.transform(self.matvec_A_box,
                                                        params, params_obj,
                                                        params_eq, params_ineq)
    pytree = self._box_osqp.l2_optimality_error(params, **hyper_params)
    return tree_l2_norm(pytree)
