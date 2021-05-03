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

"""Implementation of gradient descent in JAX."""

from typing import Any
from typing import Callable
from typing import Union

from jaxopt import implicit_diff as idf
from jaxopt import linear_solve
from jaxopt import prox
from jaxopt import proximal_gradient


def make_solver_fun(fun: Callable,
                    init: Any,
                    stepsize: float = 0.0,
                    maxiter: int = 500,
                    maxls: int = 15,
                    tol: float = 1e-3,
                    acceleration: bool = True,
                    verbose: int = 0,
                    implicit_diff: Union[bool,Callable] = True,
                    ret_info: bool = False) -> Callable:
  """Creates a gradient descent solver function ``solver_fun(params_fun)`` for
    solving::

    argmin_x fun(x, params_fun)

  where fun is smooth.

  The stopping criterion is::

    ||grad(fun)(x, params_fun)||_2 <= tol.

  Currently, this implementation is just a thin wrapper around
  ``proximal_gradient``.

  Args:
    fun: a smooth function of the form ``fun(x, params_fun)``.
    init: initialization to use for x (pytree).
    stepsize: a stepsize to use (if <= 0, use backtracking line search).
    maxiter: maximum number of proximal gradient descent iterations.
    maxls: maximum number of iterations to use in the line search.
    tol: tolerance to use.
    acceleration: whether to use acceleration or not.
    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.
    implicit_diff: if True, enable implicit differentiation using cg,
      if Callable, do implicit differentiation using callable as linear solver,
      if False, enable autodiff (this triggers loop unrolling),
    ret_info: whether to return an OptimizeResults object containing additional
      information regarding the solution

  Returns:
    Solver function ``solver_fun(params_fun)``.
  """
  pg_fun = proximal_gradient.make_solver_fun(fun=fun, prox=prox.prox_none,
                                             init=init, stepsize=stepsize,
                                             maxiter=maxiter, maxls=maxls,
                                             tol=tol, acceleration=acceleration,
                                             verbose=verbose, ret_info=ret_info,
                                             implicit_diff=implicit_diff)
  def solver_fun(params_fun):
    return pg_fun(params_fun=params_fun)

  if implicit_diff:
    if isinstance(implicit_diff, Callable):
      solve = implicit_diff
    else:
      solve = linear_solve.solve_normal_cg
    fixed_point_fun = idf.make_gradient_descent_fixed_point_fun(fun)
    solver_fun = idf.custom_fixed_point(fixed_point_fun,
                                        solve=solve)(solver_fun)

  return solver_fun
