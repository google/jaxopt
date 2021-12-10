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

"""Quadratic programming with equality constraints only."""

from typing import Any
from typing import Callable
from typing import Optional

from functools import partial
from dataclasses import dataclass

import jax

from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src.tree_util import tree_add, tree_sub
from jaxopt._src.tree_util import tree_vdot, tree_negative, tree_l2_norm
from jaxopt._src.linear_operator import _make_linear_operator
from jaxopt._src.cvxpy_wrapper import _check_params
import jaxopt._src.linear_solve as linear_solve
from jaxopt._src.iterative_refinement import IterativeRefinement


def _make_eq_qp_optimality_fun(matvec_Q, matvec_A):
  """Makes the optimality function for quadratic programming.

  Returns:
    optimality_fun(params, params_obj, params_eq, params_ineq) where
      params = (primal_var, eq_dual_var, None)
      params_obj = (params_Q, c)
      params_eq = (params_A, b)
  """

  def obj_fun(primal_var, params_obj):
    params_Q, c = params_obj
    Q = matvec_Q(params_Q)
    return 0.5 * tree_vdot(primal_var, Q(primal_var)) + tree_vdot(primal_var, c)

  def eq_fun(primal_var, params_eq):
    params_A, b = params_eq
    A = matvec_A(params_A)
    return tree_sub(A(primal_var), b)

  optimality_fun_with_ineq = idf.make_kkt_optimality_fun(obj_fun, eq_fun, ineq_fun=None)

  # It is required to post_process the output of `idf.make_kkt_optimality_fun`
  # to make the signatures of optimality_fun() and run() agree.
  def optimality_fun(params, params_obj, params_eq):
    return optimality_fun_with_ineq(params, params_obj, params_eq, None)

  return optimality_fun


@dataclass(eq=False)
class EqualityConstrainedQP(base.Solver):
  """Quadratic programming with equality constraints only.

  Supports implicit differentiation, matvec and pytrees.
  Can benefit from GPU/TPU acceleration.

  If the algorithm diverges on some instances, it might be useful to tweak the
  ``refine_regularization`` parameter.

  Attributes:
    matvec_Q: a Callable matvec_Q(params_Q, u).
      By default, matvec_Q(Q, u) = dot(Q, u), where Q = params_Q.
    matvec_A: a Callable matvec_A(params_A, u).
      By default, matvec_A(A, u) = dot(A, u), where A = params_A.
    solve: a Callable to solve linear systems, that accepts matvecs
      (default: linear_solve.solve_gmres).
    refine_regularization: a float (default: 0.) used to regularize the system.
      Useful for badly conditioned problems that lead to divergence.
      ``IterativeRefinement`` is used to correct the error introduced.
    refine_maxiter: maximum number of refinement steps, when
      refine_regularization is not 0.
    maxiter: maximum number of iterations.
    tol: tolerance (stoping criterion).
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: "auto").
  """

  matvec_Q: Optional[Callable] = None
  matvec_A: Optional[Callable] = None
  solve: Callable = linear_solve.solve_gmres
  refine_regularization: float = 0.0
  refine_maxiter: int = 10
  maxiter: int = 1000
  tol: float = 1e-5
  implicit_diff_solve: Optional[Callable] = None
  jit: bool = True

  def _refined_solve(self, matvec, b, init, maxiter, tol):
    # Instead of solving S x = b
    # We solve     \bar{S} x = b
    #
    # S = [[  Q   A^T
    #         A    0 ]]
    #
    # \bar{S} = [[ Q + delta      A^T
    #                A         - delta ]]
    #
    # Since Q is PSD, and delta a diagonal matrix,
    # this makes \bar{S} a quasi-definite matrix.
    #
    # Quasi-Definite matrices are indefinite,
    # but they are never singular (when delta > 0).
    #
    # This guarantees that the system is well posed in every case,
    # even if some constraints of A are redundant.
    #
    # The particular form of this system is inspired by [2].
    # Quasi-Definite matrices are a known tool in the literature on
    # Interior Point methods [1].
    #
    # References:
    #
    #  [1] Vanderbei, R.J., 1995. Symmetric quasidefinite matrices.
    #      SIAM Journal on Optimization, 5(1), pp.100-113.
    #
    #  [2] Stellato, B., Banjac, G., Goulart, P., Bemporad, A. and Boyd, S., 2020.
    #      OSQP: An operator splitting solver for quadratic programs.
    #      Mathematical Programming Computation, 12(4), pp.637-672.

    def matvec_qp(_, x):
      return matvec(x)

    ridge = self.refine_regularization

    def matvec_regularized_qp(_, x):
      primal, dual_eq = x
      Stop, Sbottom = matvec(x)
      return Stop + ridge * primal, Sbottom - ridge * dual_eq

    solver = IterativeRefinement(
      matvec_A=matvec_qp,
      matvec_A_bar=matvec_regularized_qp,
      solve=partial(self.solve, maxiter=maxiter, tol=tol),
      maxiter=self.refine_maxiter,
      tol=tol,
    )
    return solver.run(init_params=init, A=None, b=b)[0]

  def run(
    self,
    init_params: Optional[base.KKTSolution] = None,
    params_obj: Optional[Any] = None,
    params_eq: Optional[Any] = None,
  ) -> base.OptStep:
    """Solves 0.5 * x^T Q x + c^T x subject to Ax = b.

    This solver returns both the primal solution (x) and the dual solution.

    Args:
      init_params: ignored.
      params_obj: (Q, c) or (params_Q, c) if matvec_Q is provided.
      params_eq: (A, b) or (params_A, b) if matvec_A is provided.
    Returns:
      (params, state),  where params = (primal_var, dual_var_eq, None)
    """
    if self._check_params:
      _check_params(params_obj, params_eq)

    params_Q, c = params_obj
    params_A, b = params_eq

    Q = self.matvec_Q(params_Q)
    A = self.matvec_A(params_A)

    def matvec(u):
      primal_u, dual_u = u
      mv_A, rmv_A = A.matvec_and_rmatvec(primal_u, dual_u)
      return (tree_add(Q(primal_u), rmv_A), mv_A)

    target = (tree_negative(c), b)

    if init_params is not None:
      init_params = (init_params.primal, init_params.dual_eq)

    # Solves the following linear system:
    # [[Q A^T]  [primal_var = [-c
    #  [A 0  ]]  dual_var  ]    b]
    if self.refine_regularization == 0.0:
      primal, dual_eq = self.solve(
        matvec,
        target,
        init=init_params,
        tol=self.tol,
        maxiter=self.maxiter,
      )
    else:
      primal, dual_eq = self._refined_solve(
        matvec, target, init_params, tol=self.tol, maxiter=self.maxiter
      )

    return base.OptStep(params=base.KKTSolution(primal, dual_eq, None), state=None)

  def l2_optimality_error(
    self,
    params: base.KKTSolution,
    params_obj: Optional[Any],
    params_eq: Optional[Any],
  ):
    """Computes the L2 norm of the KKT residuals."""
    tree = self.optimality_fun(params, params_obj, params_eq)
    return tree_l2_norm(tree)

  def __post_init__(self):
    self._check_params = self.matvec_Q is None and self.matvec_A is None

    self.matvec_Q = _make_linear_operator(self.matvec_Q)
    self.matvec_A = _make_linear_operator(self.matvec_A)

    self.optimality_fun = _make_eq_qp_optimality_fun(self.matvec_Q, self.matvec_A)

    # Set up implicit diff.
    decorator = idf.custom_root(
      self.optimality_fun, has_aux=True, solve=self.implicit_diff_solve
    )
    # pylint: disable=g-missing-from-attributes
    self.run = decorator(self.run)

    if self.jit:
      self.run = jax.jit(self.run)
