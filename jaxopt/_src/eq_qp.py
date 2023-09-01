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
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src.tree_util import tree_add, tree_sub, tree_add_scalar_mul
from jaxopt._src.tree_util import tree_vdot, tree_negative, tree_l2_norm
from jaxopt._src.tree_util import tree_zeros_like
from jaxopt._src.linear_operator import _make_linear_operator
from jaxopt._src.cvxpy_wrapper import _check_params
import jaxopt._src.linear_solve as linear_solve
from jaxopt._src.iterative_refinement import IterativeRefinement


def extract_Qc_from_obj(init_params, params_obj, fun):
  """Returns (params_Q, c) parameters from params_obj.

  Args:
    init_params: KKTSolution used for the inference of the shape of primal variables.
      Mandatory when `fun` is not `None`.
    params_obj: tuple of parameters for the objective function.
    fun: objective function.

  When `fun` is `None` it retrieves the relevant informations from the tuple `params_obj`,
  whereas when `fun` is not `None` it extracts it using AutoDiff.
  """
  if fun is None:
    params_Q, c = params_obj
    return params_Q, c
  if init_params is None or init_params.primal is None:
    raise ValueError("init_params must be provided when fun is not None.")
  zeros = tree_zeros_like(init_params.primal)
  # f(x) := 0.5 * x^T Q x + c^T x + cste
  value_and_grad_fun = jax.value_and_grad(fun, argnums=0)
  # nabla_x f(x) = Q x + c
  # f(0)         = cste
  # Q is never computed explicitly, only its action on vectors is needed
  cste, c = value_and_grad_fun(zeros, params_obj)
  return (params_obj, c, cste), c


def _make_eq_qp_optimality_fun(matvec_Q, matvec_A, fun):
  """Makes the optimality function for quadratic programming.

  Args:
    matvec_Q: a function that computes the matrix-vector product with Q.
    matvec_A: a function that computes the matrix-vector product with A.
    fun: a function that computes the objective value.

  Returns:
    optimality_fun(params, params_obj, params_eq, params_ineq) where
      params = (primal_var, eq_dual_var, None)
      params_obj = (params_Q, c)
      params_eq = (params_A, b)
  """

  if fun is None:
    def obj_fun(primal_var, params_obj):
      params_Q, c = extract_Qc_from_obj(primal_var, params_obj, fun)
      Q = matvec_Q(params_Q)
      xQx = tree_vdot(primal_var, Q(primal_var))
      cx = tree_vdot(primal_var, c)
      return tree_add_scalar_mul(cx, 0.5, xQx)
  else:
    obj_fun = fun

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

  Supports implicit differentiation, matvec, pytrees, and quadratic functions.
  Can benefit from GPU/TPU acceleration.

  If the algorithm diverges on some instances, it might be useful to tweak the
  ``refine_regularization`` parameter.

  Attributes:
    matvec_Q: a Callable matvec_Q(params_Q, u).
      By default, matvec_Q(Q, u) = dot(Q, u), where Q = params_Q.
      ``matvec_Q`` incompatible with the specification of ``fun``.
      The shape of primal variables may be inferred from params_obj = (matvec_Q, c).
    matvec_A: a Callable matvec_A(params_A, u).
      By default, matvec_A(A, u) = dot(A, u), where A = params_A.
    fun: (optional) a function with signature fun(params, params_obj) that is promised
      to be a quadratic polynomial convex with respect to params, i.e fun can be written ::
        fun(x, params_obj) = 0.5*jnp.dot(x, jnp.dot(Q, x)) + jnp.dot(c, x) + cste
      with params_obj a pytree that contains the parameters of the objective function.
      (Q, c) do not need to be explicited in params_obj by the user: c will be inferred by Jaxopt,
        and the operator x -> Qx will be computed upon request.
      ``fun`` incompatible with the specification of ``matvec_Q``.
      Note that the shape of primal cannot be inferred from params_obj anymore,
      so the user should provide it in init_params.
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
  fun: Optional[Callable] = None
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
      a = tree_add_scalar_mul(Stop, ridge, primal)
      b = tree_add_scalar_mul(Sbottom, -ridge, dual_eq)
      return a, b

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
      init_params: (optional), used to infer the shape of primal variables.
        Mandatory if ``fun`` is not None, since the shape of primal variables
        cannot be inferred from ``params_obj``
      params_obj: parameters of the quadratic objective, can be:
        a tuple (Q, c) with Q a pytree of matrices,
        or a tuple (params_Q, c) if ``matvec_Q`` is provided,
        or an arbitrary pytree if ``fun`` is provided.
      params_eq: parameters of the equality constraints, can be:
        a tuple (A, b) with A a pytree of matrices,
        or a tuple (params_A, b) if matvec_A is provided.
    Returns:
      (params, state),  where params = (primal_var, dual_var_eq, None)
    """
    if self._check_params and False:
      _check_params(params_obj, params_eq)

    params_Q, c = extract_Qc_from_obj(init_params, params_obj, self.fun)
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

    return base.OptStep(params=base.KKTSolution(primal, dual_eq, None),
                        state=None)

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

    self.optimality_fun = _make_eq_qp_optimality_fun(self.matvec_Q,
                                                     self.matvec_A,
                                                     self.fun)

    # Set up implicit diff.
    decorator = idf.custom_root(
      self.optimality_fun, has_aux=True, solve=self.implicit_diff_solve
    )
    # pylint: disable=g-missing-from-attributes
    self.run = decorator(self.run)

    if self.jit:
      self.run = jax.jit(self.run)
