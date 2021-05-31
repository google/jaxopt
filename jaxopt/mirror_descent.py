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

"""Implementation of mirror descent in JAX."""

from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import jax
import jax.numpy as jnp

from jaxopt import base
from jaxopt import implicit_diff as idf
from jaxopt import linear_solve
from jaxopt import loop
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_sub


def _make_projection_grad(projection, mapping_fun, params_proj):
  """Makes the update function:

    projection(mapping_fun(curr_x) - curr_stepsize * curr_x_fun_grad,
               params_proj)
  """

  def projection_grad(curr_x, curr_x_fun_grad, curr_stepsize):
    update = tree_add_scalar_mul(
        mapping_fun(curr_x), -curr_stepsize, curr_x_fun_grad)
    return projection(update, params_proj)

  return projection_grad


def _make_md_body_fun(fun: Callable,
                      params_fun: Optional[Any] = None,
                      projection: Optional[Callable] = None,
                      mapping_fun: Optional[Callable] = None,
                      params_proj: Optional[Any] = None,
                      stepsize: Union[float, Callable] = 1.0) -> Callable:
  """Creates a body_fun for performing one iteration of mirror descent."""

  fun = jax.jit(fun)
  grad_fun = jax.jit(jax.grad(fun))
  projection_grad = _make_projection_grad(projection, mapping_fun, params_proj)

  def error_fun(curr_x, curr_x_fun_grad):
    next_x = projection_grad(curr_x, curr_x_fun_grad, 1.0)
    diff_x = tree_sub(next_x, curr_x)
    return tree_l2_norm(diff_x)

  def body_fun_mirror_descent(args):
    iter_num, curr_x, _ = args
    curr_x_fun_grad = grad_fun(curr_x, params_fun)
    curr_stepsize = (stepsize(iter_num) if isinstance(stepsize, Callable)
                     else stepsize)
    next_x = projection_grad(curr_x, curr_x_fun_grad, curr_stepsize)
    curr_error = error_fun(curr_x, curr_x_fun_grad)
    return iter_num + 1, next_x, curr_error

  return body_fun_mirror_descent


def _mirror_descent(fun, init, params_fun, projection, mapping_fun, params_proj,
                    stepsize, maxiter, tol, verbose, implicit_diff, ret_info):

  def cond_fun(args):
    iter_num = args[0]
    error = args[-1]
    if verbose:
      print(iter_num, error)
    return error > tol

  body_fun = _make_md_body_fun(
      fun=fun,
      params_fun=params_fun,
      projection=projection,
      mapping_fun=mapping_fun,
      params_proj=params_proj,
      stepsize=stepsize)

  # iter_num, curr_x, error
  args = (0, init, jnp.inf)
  # We always jit unless verbose mode is enabled.
  jit = not verbose
  # We unroll when implicit diff is disabled or when jit is disabled.
  unroll = not implicit_diff or not jit

  res = loop.while_loop(
      cond_fun=cond_fun,
      body_fun=body_fun,
      init_val=args,
      maxiter=maxiter,
      unroll=unroll,
      jit=jit)

  if ret_info:
    return base.OptimizeResults(x=res[1], nit=res[0], error=res[-1])
  else:
    return res[1]


def make_solver_fun(fun: Callable,
                    projection: Callable,
                    mapping_fun: Callable,
                    init: Any,
                    stepsize: Union[float, Callable],
                    maxiter: int = 500,
                    tol: float = 1e-2,
                    verbose: int = 0,
                    implicit_diff: Union[bool, Callable] = True,
                    ret_info: bool = False,
                    has_aux: bool = False) -> Callable:
  """Creates a mirror descent solver function
  ``solver_fun(params_fun, params_proj)`` for solving::

    argmin_x fun(x, params_fun),

  where fun is smooth with convex domain.

  The stopping criterion is::

    ||x - projection(mapping_fun(x) - g, params_proj)||_2 <= tol,
  where ``g = grad(fun)(x, params_fun)``.

  Args:
    fun: a smooth function of the form ``fun(x, params_fun)``.
    projection: projection operator of the form ``projection(x, params_proj)``,
      typically ``argmin_z D_{gen_fun}(z, mapping_fun^{-1}(y))``.
    mapping_fun: the mirror map, typically of the form
      ``mapping_fun = grad(gen_fun)``, where `gen_fun` is the generating
      function of the Bregman divergence.
    init: initialization to use for x (pytree).
    stepsize: a stepsize to use, or a callable specifying the stepsize to use at
      each iteration.
    maxiter: maximum number of mirror descent iterations.
    tol: tolerance to use.
    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.
    implicit_diff: if True, enable implicit differentiation using cg,
      if Callable, do implicit differentiation using callable as linear solver,
      if False, use autodiff through the solver implementation (note:
        this will unroll syntactic loops).
    ret_info: whether to return an OptimizeResults object containing additional
      information regarding the solution
    has_aux: whether function fun outputs one (False) or more values (True).
      When True it will be assumed by default that fun(...)[0] is the objective.


  Returns:
    Solver function ``solver_fun(params_fun, params_proj)``.

  References:
    Nemirovskij, Arkadij Semenovič, and David Borisovich Yudin. "Problem
    complexity and method efficiency in optimization." J. Wiley @ Sons, New
    York(1983).
  """
  _fun = fun if not has_aux else lambda x, par: fun(x, par)[0]
  def solver_fun(params_fun=None, params_proj=None):
    return _mirror_descent(_fun, init, params_fun, projection, mapping_fun,
                           params_proj, stepsize, maxiter, tol, verbose,
                           implicit_diff, ret_info)

  if implicit_diff:
    if isinstance(implicit_diff, Callable):
      solve = implicit_diff
    else:
      solve = linear_solve.solve_normal_cg
    fixed_point_fun = idf.make_mirror_descent_fixed_point_fun(_fun,
                                                              projection,
                                                              mapping_fun)
    solver_fun = idf.custom_fixed_point(fixed_point_fun,
                                        unpack_params=True,
                                        solve=solve)(solver_fun)
  return solver_fun
