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
"""Iterative Refinement for linear systems."""

# TODO(lbethune): in the future Chebychev acceleration of Iterative Refinement
# could be implemented:
#   Arioli, M. and Scott, J., 2014. Chebyshev acceleration of iterative refinement.
#   Numerical Algorithms, 66(3), pp.591-608.

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp 

from jaxopt._src import loop
from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src.tree_util import tree_zeros_like, tree_add, tree_sub
from jaxopt._src.tree_util import tree_add_scalar_mul, tree_scalar_mul 
from jaxopt._src.tree_util import tree_vdot, tree_negative, tree_l2_norm
from jaxopt._src.linear_operator import _make_linear_operator
import jaxopt._src.linear_solve as linear_solve


class IterativeRefinementState(NamedTuple):
  """Named tuple containing state information.

  Attributes:
    iter_num: iteration number.
    error: error used as stop criterion, deduced from residuals b - Ax.
    target_residuals: residuals of the current target.
    init: (optional) params to warm start the inner solver at next iteration.
  """
  iter_num: int
  error: float
  target_residuals: Any
  init: Optional[Any]
  # TODO(lbethune): in the future return the state of the internal
  # solver (iter_num, error) as part of the current state.


@dataclass(eq=False)
class IterativeRefinement(base.IterativeSolver):
  """Iterativement refinement to solve linear systems.

  Solves Ax = b up to high accuracy with low precision solver.

  The algorithm starts with :math:`r_0, x_0 = b, 0` and solves iteratively::

    \begin{aligned}
    \text{solve for x}  \bar{A} x &= r_{t-1}\\
                        x_t &= x_{t-1} + x\\
                        r_t &= b - A x_t
    \end{aligned}
  where :math:`\bar{A}` is some approximation of A, with preferably
  better preconditonning than A.

  This method has the advantage to converge even if the solve step is inaccurate.  
  This is particularly useful for ill-posed problems, with strong regularization on
  :math:`\bar{A}`, in ``float32`` environment, or with solvers that struggle to converge.

  Attributes:
    solve: a Callable that accepts A as first argument, b as second,
      and a warm start ``init`` as third argument.
      This solver can be inaccurate and run with low precision.
    matvec: (optional) a Callable matvec_A(params_A, x).
      By default, matvec_A(A, x) = tree_dot(A, x), where tree pytree A = params_A matches x structure.
    matvec_A_bar: (optional) a Callable.
      If None, then :math:`\bar{A}=A`. Otherwise, a Callable matvec_A_bar(x).
    maxiter: maximum number of iterations (default: 10).
    tol: absolute tolerance for stoping criterion (default: 1e-7).
    verbose: If verbose=1, print error at each iteration.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto")

  References:
    
    [1] J. H. Wilkinson. Rounding Errors in Algebraic Processes. Prentice Hall, Englewood Cliffs, NJ, 1963.  
      
    [2] Moler, C.B., 1967. Iterative refinement in floating point. Journal of the ACM (JACM), 14(2), pp.316-321.  
      
    [3] Wikipedia contributors, "Iterative refinement,"
        Wikipedia, The Free Encyclopedia,
        https://en.wikipedia.org/w/index.php?title=Iterative_refinement&oldid=1055633064 (accessed December 3, 2021).
  """
  matvec_A: Optional[Callable] = None
  matvec_A_bar: Optional[Callable] = None
  solve: Callable = partial(linear_solve.solve_gmres, ridge=1e-6)
  maxiter: int = 10
  tol: float = 1e-7
  verbose: int = 0
  implicit_diff_solve: Optional[Callable] = None
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self,
                 init_params: Any,
                 params_A: Any,
                 b: Any,
                 params_A_bar: Any = None):
    return IterativeRefinementState(
      iter_num=jnp.asarray(0),
      error=jnp.asarray(jnp.inf),
      target_residuals=b,
      init=init_params)

  def init_params(self,
                  params_A: Any,
                  b: Any,
                  params_A_bar: Any = None):
    return tree_zeros_like(b)

  def update(self,
             params: Any,
             state: IterativeRefinementState,
             params_A: Any,
             b: Any,
             params_A_bar: Optional[Any] = None):
    if self._copy_params_A:
      params_A_bar = params_A

    A = self.matvec_A(params_A)
    A_bar = self.matvec_A_bar(params_A_bar)

    # TODO(lbethune): support preconditioners ?
    # Could it be done by user with partial(solver, M=precond) ?
    residual_sol = self.solve(A_bar, state.target_residuals, init=state.init)

    params = tree_add(params, residual_sol)

    target_residuals = tree_sub(b, A(params))
    error = tree_l2_norm(target_residuals)

    state = IterativeRefinementState(
      iter_num=state.iter_num+1,
      error=error,
      target_residuals=target_residuals,
      init=None)
    
    return base.OptStep(params, state)

  def run(self,
          init_params: Optional[Any],
          params_A: Any,
          b: Any,
          params_A_bar: Optional[Any] = None):
    if init_params is None:
      init_params = self.init_params(params_A, b, params_A_bar)
    return super().run(init_params, params_A, b, params_A_bar)
    
  def optimality_fun(self,
                     params: Any,
                     params_A: Any,
                     b: Any,
                     params_A_bar: Optional[Any] = None):
    del params_A_bar  # unused
    A = self.matvec_A(params_A)
    return tree_sub(b, A(params))

  def l2_optimality_error(self,
                          params: Any,
                          params_A: Any,
                          b: Any,
                          params_A_bar: Optional[Any] = None):
    del params_A_bar  # unused
    return tree_l2_norm(self.optimality_fun(params, params_A, b))

  def __post_init__(self):
    self._copy_params_A = False
    if self.matvec_A_bar is None:
      self.matvec_A_bar = self.matvec_A
      self._copy_params_A = True
    
    self.matvec_A = _make_linear_operator(self.matvec_A)
    self.matvec_A_bar = _make_linear_operator(self.matvec_A_bar)


def solve_iterative_refinement(matvec: Callable,
                               b: Any,
                               init: Optional[Any] = None,
                               maxiter: int = 10,
                               tol: float = 1e-7,
                               solve: Callable = linear_solve.solve_gmres,
                               **kwargs) -> Any:
  """Solves ``A x = b`` using iterative refinement.

  Args:
    matvec: product between ``A`` and a vector.
    b: pytree.
    maxiter: maximum number of refinement steps (default: 10).
    tol: absolute tolerance residuals (default: 1e-7).
    solve: optional solve function (default: linear_solve.solve_gmres).
    kwargs: additional parameters for IterativeRefinement.

  Returns:
    Pytree with the same structure as ``b``.
  """
  iterative_refinement = IterativeRefinement(matvec_A=lambda _,x: matvec(x),
                                             solve=solve,
                                             maxiter=maxiter,
                                             tol=tol,
                                             **kwargs)
  return iterative_refinement.run(init, params_A=None, b=b)[0]

