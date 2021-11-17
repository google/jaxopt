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

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from dataclasses import dataclass
from functools import partial

import jax
import jax.nn as nn
import jax.numpy as jnp
from jax.tree_util import tree_reduce

from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src.tree_util import tree_add, tree_sub, tree_mul, tree_div 
from jaxopt._src.tree_util import tree_scalar_mul, tree_add_scalar_mul, tree_mean
from jaxopt._src.tree_util import tree_map, tree_vdot, tree_l2_norm, tree_inf_norm
from jaxopt._src.tree_util import tree_ones_like, tree_zeros_like, tree_where
from jaxopt._src.tree_util import tree_reciproqual, tree_negative
from jaxopt._src.linear_operator import DenseLinearOperator, FunctionalLinearOperator
from jaxopt._src.linear_solve import solve_cg
from jaxopt._src.quadratic_prog import _matvec_and_rmatvec
from jaxopt.projection import projection_box
from jaxopt import loop


def _make_osqp_optimality_fun(matvec_Q, matvec_A):
  """Makes the optimality function for OSQP.

  Returns:
    optimality_fun(params, params_obj, params_eq, params_ineq) where
      params = (primal_var, eq_dual_var, ineq_dual_var)
      params_obj = (P, c)
      params_eq = A
      params_ineq = (l, u)
  """
  def obj_fun(primal_var, params_obj):
    x, _ = primal_var
    params_Q, c = params_obj
    Q = matvec_Q(params_Q)
    # minimize 0.5 x^T Q x + c^T x
    qp_obj = tree_add_scalar_mul(tree_vdot(c, x), 0.5, tree_vdot(x, Q(x)))
    return qp_obj

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


class OSQPState(NamedTuple):
  """Named tuple containing state information.

  Attributes:
    iter_num: iteration number
    error: error used as stop criterion, deduced from residuals
    status: integer, one of ``[OSQP.UNSOLVED, OSQP.SOLVED, OSQP.PRIMAL_INFEASIBLE, OSQP.DUAL_INFEASIBLE]``.
    primal_residuals: residuals of constraints of primal problem
    dual_residuals: residuals of constraints of dual problem
    rho_bar: current stepsize.
    eq_qp_last_sol: solution of equality constrained QP. Useful for warm start.
    params_precond: parameters for preconditioner of equality constrained QP.
  """
  iter_num: int
  error: float
  status: int
  primal_residuals: Any
  dual_residuals: Any
  rho_bar: Any
  eq_qp_last_sol: Tuple[Any, Any]
  params_precond: Any


def _make_linear_operator(matvec):
  if matvec is None:
    return DenseLinearOperator
  else:
    return partial(FunctionalLinearOperator, matvec)


@dataclass
class OSQPPreconditioner(ABC):
  """Abstract class for OSQP preconditioners.
  
  Must approximate the inverse of Mx = Px + sigma x + rho_bar A^T A x.
  
  Attributes:
    matvec_Q: (optional) a Callable matvec_Q(params_Q, x).
    matvec_A: (optional) a Callable matvec_A(params_A, x).
  """
  matvec_Q: Optional[Callable] = None
  matvec_A: Optional[Callable] = None

  @abstractmethod
  def __call__(self, precond_params, x):
    """Computes M^{-1}x."""
    pass
  
  @abstractmethod
  def init_params_precond(self, params_Q, params_A, sigma, rho_bar):
    """Get initial parameters for preconditioner."""
    pass

  @abstractmethod
  def update_stepsize(self, precond_params, rho_bar):
    """Update preconditioner parameters when rho_bar changes; should be efficient."""
    pass


class NoPreconditioner(OSQPPreconditioner):
  """Dummy class for no-preconditioning."""

  def __call__(self, params_precond, x):
    return x
  
  def init_params_precond(self, params_Q, params_A, sigma, rho_bar):
    return None

  def update_stepsize(self, params_precond, rho_bar):
    return None


class JacobiPreconditioner(OSQPPreconditioner):
  """Jacobi preconditioner (only available for pytree of matrices)."""

  def __call__(self, params_precond, x):
    precond_Q, precond_A, sigma, rho_bar = params_precond
    diag_precond = tree_add_scalar_mul(precond_Q, rho_bar, precond_A)
    diag_precond = tree_add(diag_precond, sigma)
    inv_diag = tree_map(lambda m_diag: 1/m_diag, diag_precond)
    return tree_mul(inv_diag, x)

  def init_params_precond(self, params_Q, params_A, sigma, rho_bar):
    precond_Q = DenseLinearOperator(params_Q).diag()
    precond_A = DenseLinearOperator(params_A).columns_l2_norms(squared=True)
    return precond_Q, precond_A, sigma, rho_bar

  def update_stepsize(self, params_precond, rho_bar):
    return params_precond[:-1] + (rho_bar,)


@dataclass
class OSQP(base.IterativeSolver):
  """Operator Splitting Solver for Quadratic Programs.

  Jax implementation of the celebrated GPU-OSQP [1,3] based on ADMM.

  It solves convex problems of the form::
  
    \begin{aligned}
      \min_{x,z} \quad & \frac{1}{2}xQx + c^Tx\\
      \textrm{s.t.} \quad & Ax=z\\
        &l\leq z\leq u    \\
    \end{aligned}
  
  Equality constraints are obtained by setting l = u.
  If the inequality is one-sided then ``jnp.inf can be used for u,
  and ``-jnp.inf`` for l.

  P must be a positive semidefinite (PSD) matrix.

  The Lagrangian is given by::

    \mathcal{L} = \frac{1}{2}x^TQx + c^Tx + y^T(Ax-z) + mu^T (z-u) + phi^T (l-z)

  Primal variables: x, z
  Dual variables  : y, mu, phi

  ADMM computes y at each iteration. mu and phi can be deduced from z and y.
  Defaults values for hyper-parameters come from: https://github.com/osqp/osqp/blob/master/include/constants.h

  Attributes:
    matvec_Q: (optional) a Callable matvec_Q(params_Q, x).
      By default, matvec_Q(P, x) = tree_dot(P, x), where the pytree P = params_Q matches x structure.
    matvec_A: (optional) a Callable matvec_A(params_A, x).
      By default, matvec_A(A, x) = tree_dot(A, x), where tree pytree A = params_A matches x structure.
    check_primal_dual_infeasability: if True populates the ``status`` field of ``state``
      with one of ``OSQP.PRIMAL_INFEASIBLE``, ``OSQP.DUAL_INFEASIBLE``.
      If False it improves speed but does not check feasability.
      If the problem is primal or dual infeasible, and jit=False, then a ValueError exception is raised.
      If "auto", it will be True if jit=False and False otherwise. (default: "auto")
    sigma: ridge regularization parameter in linear system.
    momentum: relaxation parameter (default: 1.6), must belong to the open interval (0,2).
      momentum=1 => no relaxation.
      momentum<1 => under-relaxation.
      momentum>1 => over-relaxation.
      Boyd [2, p21] suggests chosing momentum in [1.5, 1.8].
    eq_qp_preconditioner: (optional) a string specifying the pre-conditioner (default: None).
    eq_qp_solve_tol: tolerance for linear solver in equality constrained QP. (default: 1e-5)
      High tolerance may speedup each ADMM step but will slow down overall convergence. 
    eq_qp_solve_maxiter: number of iterations for linear solver in equality constrained QP. (default: None)
      Low maxiter will speedup each ADMM step but may slow down overall convergence.
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
    verbose: If verbose=1, print error at each iteration. If verbose=2, also print stepsizes and primal/dual variables.
      Warning: verbose>0 will automatically disable jit.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto")

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
  check_primal_dual_infeasability: base.AutoOrBoolean = "auto"
  sigma: float = 1e-6
  momentum: float = 1.6
  eq_qp_preconditioner: Optional[str] = None
  eq_qp_solve_tol: float = 1e-7
  eq_qp_solve_maxiter: Optional[int] = None
  rho_start: float = 0.1
  rho_min: float = 1e-6
  rho_max: float = 1e6
  stepsize_updates_frequency: int = 10
  primal_infeasible_tol: float = 1e-4
  dual_infeasible_tol: float = 1e-4
  maxiter: int = 4000
  tol: float = 1e-3
  termination_check_frequency: int = 5
  verbose: int = 0
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"


  # class attributes (ignored by @dataclass)
  UNSOLVED          = 0  # stopping criterion not reached yet.
  SOLVED            = 1  # feasible solution found with satisfying precision.
  DUAL_INFEASIBLE   = 2  # infeasible dual (infeasible primal or unbounded primal).
  PRIMAL_INFEASIBLE = 3  # infeasible primal.

  def init_state(self, init_params, params_obj, params_eq, params_ineq):
    x, z = init_params.primal
    y    = init_params.dual_eq
    Q    = self.matvec_Q(params_obj[0])
    c    = params_obj[1]
    A    = self.matvec_A(params_eq)

    primal_residuals, dual_residuals = self._compute_residuals(Q, c, A, x, z, y)
    params_precond = self._eq_qp_preconditioner_impl.init_params_precond(params_obj[0], params_obj[1],
                                                                         self.sigma, self.rho_start)

    return OSQPState(iter_num=0,
                     error=jnp.inf,
                     status=OSQP.UNSOLVED,
                     primal_residuals=primal_residuals,
                     dual_residuals=dual_residuals,
                     rho_bar=self.rho_start,
                     eq_qp_last_sol=x,
                     params_precond=params_precond)
  
  def init_params(self, init_x, params_obj, params_eq, params_ineq):
    """Return defaults params for initialization."""
    if init_x is None:
      init_x = tree_zeros_like(params_obj[1])
    init_z = self.matvec_A(params_eq)(init_x)
    init_y = tree_zeros_like(init_z)
    return base.KKTSolution((init_x, init_z), init_y, (init_y, init_y))

  def _get_full_KKT_solution(primal, y):
    """Returns all dual variables of the problem."""
    # Unfortunately OSQP algorithm only returns y as dual variable,
    # mu and phi are missing, but can be recovered:
    #
    # We distinguish between l=u and l<u.
    # If l<u there are three cases:
    #   1. l < z < u: phi=0  mu=0 (and y=0)
    #   2. l = z < u: phi=-y mu=0 (and y<0)
    #   3. l < z = u: phi=0  mu=y (and y>0)
    #  this can be simplified with mu=relu(y) and phi=relu(-y)
    # If l=u then y=mu-phi then we have one degree of liberty to chose mu and phi
    # by symmetry with previous case we may chose mu=relu(y) and phi=relu(-y).
    is_pos = tree_map(lambda yi: yi >= 0, y)
    mu  = tree_where(is_pos, y, 0)  # derivative = 1 in y = 0
    phi = tree_map(lambda yi: jax.nn.relu(-yi), y)  # derivative = 0 in y = 0
    # y = mu - phi
    # d_y = d_mu - d_phi = 1 (everywhere; including in zero)
    return base.KKTSolution(primal=primal, dual_eq=y, dual_ineq=(mu, phi))

  def _update_stepsize(self, rho_bar, params_precond, primal_residuals, dual_residuals, Q, c, A, x, y):
    """Update stepsize based on the ratio between primal and dual residuals."""
    Ax, ATy     = A.matvec_and_rmatvec(x, y)
    primal_coef = tree_inf_norm(primal_residuals) / tree_inf_norm(Ax)
    max_inf     = jnp.maximum(tree_inf_norm(Q(x)), jnp.maximum(tree_inf_norm(ATy), tree_inf_norm(c)))
    dual_coef   = tree_inf_norm(dual_residuals) / max_inf
    eps_div     = jnp.finfo(dual_coef.dtype).eps
    coef        = jnp.sqrt(primal_coef / (dual_coef + eps_div))
    rho_bar     = jnp.clip(rho_bar * coef, self.rho_min, self.rho_max)
    params_precond = self._eq_qp_preconditioner_impl.update_stepsize(params_precond, rho_bar)
    return rho_bar, params_precond

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
    status = jnp.where(criterion <= self.tol, OSQP.SOLVED, OSQP.UNSOLVED)
    return criterion, status

  def _check_dual_infeasability(self, error, status, delta_x, Q, c, Adx, l, u):
    criterion  = self.dual_infeasible_tol * tree_inf_norm(delta_x)

    certif_Q   = tree_inf_norm(Q(delta_x))
    certif_c   = tree_vdot(c, delta_x)

    unbouned_l = tree_map(lambda li: li == -jnp.inf, l)
    unbouned_u = tree_map(lambda ui: ui == jnp.inf, u)
    certif_l   = tree_map(lambda adxi,li: jnp.all(li <= adxi), Adx, tree_where(unbouned_l, -jnp.inf, -criterion))
    certif_u   = tree_map(lambda adxi,ui: jnp.all(adxi <= ui), Adx, tree_where(unbouned_u, jnp.inf, criterion))
    certif_A   = tree_reduce(jnp.logical_and, tree_map(jnp.logical_and, certif_l, certif_u))

    certif_dual_infeasible = jnp.logical_and(jnp.logical_and(certif_Q <= criterion, certif_c <= criterion), certif_A)

    if self.verbose >= 2:
      print(f"certif_Q={certif_Q} certif_c={certif_c} certif_A={certif_A} criterion={criterion}, Adx={Adx}, certif_l={certif_l}, certif_u={certif_u}")

    # infeasible dual implies either infeasible primal, either unbounded primal.
    return jax.lax.cond(certif_dual_infeasible,
      lambda _: (0., OSQP.DUAL_INFEASIBLE),  # dual unfeasible; exit the main loop with error = 0.
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

    if self.verbose >= 2:
      print(f"certif_A={certif_A}, certif_lu={certif_lu}, criterion={criterion}")

    return jax.lax.cond(certif_primal_infeasible,
      lambda _: (0.,  # primal unfeasible; exit the main loop with error = 0.
                OSQP.PRIMAL_INFEASIBLE),  
      lambda _: (error, status),  # primal feasible or unbounded (depends of dual feasability).
      operand=None)  

  def _check_infeasability(self, prev_sol, sol, error, status, Q, c, A, l, u):
    delta_x = tree_sub(sol.primal[0], prev_sol.primal[0])
    delta_y = tree_sub(sol.dual_eq, prev_sol.dual_eq)
    Adx, ATdy = A.matvec_and_rmatvec(delta_x, delta_y)

    error, status = self._check_dual_infeasability(error, status, delta_x, Q, c, Adx, l, u)
    error, status = self._check_primal_infeasability(error, status, delta_y, ATdy, l, u)

    jit, _ = self._get_loop_options()
    if not jit:
      if status == OSQP.PRIMAL_INFEASIBLE:
        raise ValueError(f"Primal infeasible. Certificate: y(t+1)-y(t) = {delta_y}.")
      if status == OSQP.DUAL_INFEASIBLE:
        raise ValueError(f"Dual infeasible. Certificate: x(t+1)-x(t) = {delta_x}.")

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

    def matvec(x_bar):
      Qx_sigmax = tree_add_scalar_mul(Q(x_bar), self.sigma, x_bar)
      ATAx = A.normal_matvec(x_bar)
      return tree_add_scalar_mul(Qx_sigmax, rho_bar, ATAx)
    
    bxq = tree_sub(tree_scalar_mul(self.sigma, x), c)
    byz = tree_add_scalar_mul(y, -rho_bar, z)
    b = tree_sub(bxq, A.rmatvec(bxq, byz))

    primal_res_inf = tree_inf_norm(state.primal_residuals)
    dual_res_inf = tree_inf_norm(state.dual_residuals)
    atol = 0.15 * jnp.sqrt(primal_res_inf * dual_res_inf)
    # 0.15 < 1 implies that atol is slower than geometric mean of primal_res_inf and dual_res_inf.

    x_bar = solve_cg(matvec, b,
                     init=state.eq_qp_last_sol,
                     M=lambda x: self._eq_qp_preconditioner_impl(state.params_precond, x),
                     maxiter=self.eq_qp_solve_maxiter,
                     atol=atol,
                     tol=self.eq_qp_solve_tol * self.tol)  # adaptive threshold.
    
    return x_bar

  def _admm_step(self, params, Q, c, A, box, rho_bar, state):
    """Performs one atomic step of the ADMM algorithm."""
    x, z = params.primal
    y    = params.dual_eq  # dual variables for constraints z_bar = z;
    # mu, phi = params.dual_ineq are unused

    # lines are numbered according to the pseudo-code in the paper OSQP: https://arxiv.org/pdf/1711.08013.pdf

    # line 3: optimization step for (x_bar, z_bar)
    # this equality constrained QP is solved by writing KKT conditions
    # which reduce to a well-posed linear system.
    x_bar = self._solve_linear_system(params, Q, c, A, rho_bar, state)
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

    return (x_next, z_next), y_next, x_bar

  def update(self, params, state, params_obj, params_eq, params_ineq):
    """Perform OSQP step."""
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
    Q    = self.matvec_Q(params_obj[0])
    c    = params_obj[1]
    A    = self.matvec_A(params_eq)
    l, u = params_ineq

    # for active constraints (in particular equality constraints) high stepsize is better
    rho_bar = state.rho_bar
    if self.verbose >= 2:
      print(f"rho_bar={rho_bar}")

    (x, z), y, eq_qp_last_sol = self._admm_step(params, Q, c, A, (l, u), rho_bar, state)
    if self.verbose >= 3:
      print(f"x={x}\nz={z}\ny={y}")

    primal_residuals, dual_residuals = self._compute_residuals(Q, c, A, x, z, y)
    if self.verbose >= 3:
      print(f"primal_residuals={primal_residuals}, dual_residuals={dual_residuals}")

    rho_bar, params_precond = jax.lax.cond(jnp.mod(state.iter_num, self.stepsize_updates_frequency) == 0,
      lambda _: self._update_stepsize(rho_bar, state.params_precond, primal_residuals, dual_residuals, Q, c, A, x, y),
      lambda _: (rho_bar, state.params_precond), operand=None)

    sol = OSQP._get_full_KKT_solution(primal=(x, z), y=y)

    error, status = jax.lax.cond(jnp.mod(state.iter_num, self.termination_check_frequency) == 0,
      lambda _: self._check_termination_conditions(primal_residuals, dual_residuals,
                                                   params, sol, Q, c, A, l, u),
      lambda s: (state.error, s), operand=(state.status))

    state = OSQPState(iter_num=state.iter_num+1,
                      error=error,
                      status=status,
                      primal_residuals=primal_residuals,
                      dual_residuals=dual_residuals,
                      rho_bar=rho_bar,
                      eq_qp_last_sol=eq_qp_last_sol,
                      params_precond=params_precond)
    return base.OptStep(params=sol, state=state)

  def run(self, init_params, params_obj, params_eq, params_ineq) -> base.OptStep:
    """Return primal/dual variables.
    
    Args:
      init_params: initial KKTSolution (can be None).
      params_obj: pair (params_Q, c).
      params_eq: params_A.
      params_ineq: pair (l, u).
    """
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
    if self.eq_qp_preconditioner is None:
      self._eq_qp_preconditioner_impl = NoPreconditioner()
    elif self.eq_qp_preconditioner == 'jacobi':
      if self.matvec_Q is not None or self.matvec_A is not None:
        raise ValueError("'jacobi' preconditioner is only available for pytree of matrices (no support for matvec).")
      self._eq_qp_preconditioner_impl = JacobiPreconditioner()
    else:
      raise ValueError(f"Invalid argument eq_qp_preconditioner={self.eq_qp_preconditioner}, expected None or 'jacobi'.")

    self.matvec_Q = _make_linear_operator(self.matvec_Q)
    self.matvec_A = _make_linear_operator(self.matvec_A)
    
    if self.check_primal_dual_infeasability == "auto":
      jit, _ = self._get_loop_options()
      self.check_primal_dual_infeasability = not jit

    self.optimality_fun = _make_osqp_optimality_fun(self.matvec_Q, self.matvec_A)
