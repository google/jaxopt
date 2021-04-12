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
"""Implementation of proximal gradient descent in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

import jax
import jax.numpy as jnp

from jaxopt import implicit_diff
from jaxopt import loop
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_vdot


class OptimizeResults(NamedTuple):
  error: float
  nit: int
  x: Any


def _make_prox_grad(prox, params_prox):
  """Make the update function:

    prox(curr_x - curr_stepsize * curr_x_fun_grad, params_prox, curr_stepsize)
  """

  def prox_grad(curr_x, curr_x_fun_grad, curr_stepsize):
    update = tree_add_scalar_mul(curr_x, -curr_stepsize, curr_x_fun_grad)
    return prox(update, params_prox, curr_stepsize)

  return prox_grad


def _make_linesearch(fun, params_fun, prox_grad, maxls, stepfactor, unroll):
  """Make the backtracking line search."""

  # Currently, we never jit when unrolling, since jitting a huge graph is slow.
  # In the future, we will improve loop.while_loop similarly to
  # https://github.com/google-research/ott/blob/master/ott/core/fixed_point_loop.py
  jit = not unroll

  def linesearch(curr_x, curr_x_fun_val, curr_x_fun_grad, curr_stepsize):
    # epsilon of current dtype for robust checking of
    # sufficient decrease condition
    eps = jnp.finfo(curr_x_fun_val.dtype).eps

    def cond_fun(args):
      next_x, stepsize = args
      diff_x = tree_sub(next_x, curr_x)
      sqdist = tree_l2_norm(diff_x, squared=True)
      # the expression below checks the sufficient decrease condition
      # f(next_x) < f(x) + dot(grad_f(x), diff_x) + (0.5/stepsize) ||diff_x||^2
      # where the terms have been reordered for numerical stability
      fun_decrease = stepsize * (fun(next_x, params_fun) - curr_x_fun_val)
      condition = stepsize * tree_vdot(diff_x, curr_x_fun_grad) + 0.5 * sqdist
      return fun_decrease > condition + eps

    def body_fun(args):
      stepsize = args[1]
      next_stepsize = stepsize * stepfactor
      next_x = prox_grad(curr_x, curr_x_fun_grad, next_stepsize)
      return next_x, next_stepsize

    init_x = prox_grad(curr_x, curr_x_fun_grad, curr_stepsize)
    init_val = (init_x, curr_stepsize)

    return loop.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=init_val,
        maxiter=maxls,
        unroll=unroll,
        jit=jit)

  return linesearch


def make_proximal_gradient_body_fun(fun: Callable,
                                    params_fun: Optional[Any] = None,
                                    prox: Optional[Callable] = None,
                                    params_prox: Optional[Any] = None,
                                    stepsize: float = 0.0,
                                    maxls: int = 15,
                                    acceleration: bool = True,
                                    unroll_ls: bool = False,
                                    stepfactor: float = 0.5) -> Callable:
  """Create a body_fun for performing one iteration of proximal gradient."""

  if prox is None:
    prox = lambda x, params, scaling=1.0: x

  fun = jax.jit(fun)
  value_and_grad_fun = jax.jit(jax.value_and_grad(fun))
  grad_fun = jax.jit(jax.grad(fun))
  prox_grad = _make_prox_grad(prox, params_prox)
  linesearch = _make_linesearch(
      fun=fun,
      params_fun=params_fun,
      prox_grad=prox_grad,
      maxls=maxls,
      stepfactor=stepfactor,
      unroll=unroll_ls)

  def error_fun(curr_x, curr_x_fun_grad):
    next_x = prox_grad(curr_x, curr_x_fun_grad, 1.0)
    diff_x = tree_sub(next_x, curr_x)
    return tree_l2_norm(diff_x)

  def _iter(curr_x, curr_x_fun_val, curr_x_fun_grad, curr_stepsize):
    if stepsize <= 0:
      # With line search.
      next_x, next_stepsize = linesearch(curr_x, curr_x_fun_val,
                                         curr_x_fun_grad, curr_stepsize)

      # If step size becomes too small, we restart it to 1.0.
      # Otherwise, we attempt to increase it.
      next_stepsize = jnp.where(next_stepsize <= 1e-6, 1.0,
                                next_stepsize / stepfactor)

      return next_x, next_stepsize
    else:
      # Without line search.
      next_x = prox_grad(curr_x, curr_x_fun_grad, stepsize)
      return next_x, stepsize

  def body_fun_proximal_gradient(args):
    iter_num, curr_x, curr_stepsize, _ = args
    curr_x_fun_val, curr_x_fun_grad = value_and_grad_fun(curr_x, params_fun)
    next_x, next_stepsize = _iter(curr_x, curr_x_fun_val, curr_x_fun_grad,
                                  curr_stepsize)
    curr_error = error_fun(curr_x, curr_x_fun_grad)
    return iter_num + 1, next_x, next_stepsize, curr_error

  def body_fun_accelerated_proximal_gradient(args):
    iter_num, curr_x, curr_y, curr_t, curr_stepsize, _ = args
    curr_y_fun_val, curr_y_fun_grad = value_and_grad_fun(curr_y, params_fun)
    next_x, next_stepsize = _iter(curr_y, curr_y_fun_val, curr_y_fun_grad,
                                  curr_stepsize)
    next_t = 0.5 * (1 + jnp.sqrt(1 + 4 * curr_t**2))
    diff_x = tree_sub(next_x, curr_x)
    next_y = tree_add_scalar_mul(next_x, (curr_t - 1) / next_t, diff_x)
    next_x_fun_grad = grad_fun(next_x, params_fun)
    next_error = error_fun(next_x, next_x_fun_grad)
    return iter_num + 1, next_x, next_y, next_t, next_stepsize, next_error

  if acceleration:
    return body_fun_accelerated_proximal_gradient
  else:
    return body_fun_proximal_gradient


def _proximal_gradient(fun, init, params_fun, prox, params_prox, stepsize,
                       maxiter, maxls, tol, acceleration, verbose, unroll,
                       ret_info):

  def cond_fun(args):
    iter_num = args[0]
    error = args[-1]
    if verbose:
      print(iter_num, error)
    return error > tol

  body_fun = make_proximal_gradient_body_fun(
      fun=fun,
      params_fun=params_fun,
      prox=prox,
      params_prox=params_prox,
      stepsize=stepsize,
      maxls=maxls,
      acceleration=acceleration,
      unroll_ls=unroll)

  if acceleration:
    # iter_num, curr_x, curr_y, curr_t, curr_stepsize, error
    args = (0, init, init, 1.0, 1.0, 1e6)
  else:
    # iter_num, curr_x, curr_stepsize, error
    args = (0, init, 1.0, 1e6)

  # Currently, we always unroll in verbose mode.
  unroll = unroll or verbose

  # Currently, we never jit when unrolling, since jitting a huge graph is slow.
  # In the future, we will improve loop.while_loop similarly to
  # https://github.com/google-research/ott/blob/master/ott/core/fixed_point_loop.py
  jit = not unroll

  res = loop.while_loop(
      cond_fun=cond_fun,
      body_fun=body_fun,
      init_val=args,
      maxiter=maxiter,
      unroll=unroll,
      jit=jit)

  if ret_info:
    return OptimizeResults(x=res[1], nit=res[0], error=res[-1])
  else:
    return res[1]


def _proximal_gradient_fwd(fun, init, params_fun, prox, params_prox, stepsize,
                           maxiter, maxls, tol, acceleration, verbose, unroll,
                           ret_info):
  sol = _proximal_gradient(fun, init, params_fun, prox, params_prox, stepsize,
                           maxiter, maxls, tol, acceleration, verbose, unroll,
                           ret_info)
  return sol, (params_fun, params_prox, sol)


def _proximal_gradient_bwd(fun, prox, stepsize, maxiter, maxls, tol,
                           acceleration, verbose, unroll, ret_info, res,
                           cotangent):
  params_fun, params_prox, sol = res
  if prox is None:
    vjp = implicit_diff.gd_fixed_point_vjp(
        fun=fun, sol=sol, params_fun=params_fun, cotangent=cotangent)
    return (None, vjp, None)
  else:
    vjps = implicit_diff.pg_fixed_point_vjp(
        fun=fun,
        sol=sol,
        params_fun=params_fun,
        prox=prox,
        params_prox=params_prox,
        cotangent=cotangent)
    return (None, vjps[0], vjps[1])


def proximal_gradient(fun: Callable,
                      init: Any,
                      params_fun: Optional[Any] = None,
                      prox: Optional[Callable] = None,
                      params_prox: Optional[Any] = None,
                      stepsize: float = 0.0,
                      maxiter: int = 500,
                      maxls: int = 15,
                      tol: float = 1e-3,
                      acceleration: bool = True,
                      verbose: int = 0,
                      implicit_diff: bool = True,
                      ret_info: bool = False) -> Any:
  """Solves argmin_x fun(x, params_fun) + g(x, params_prox),

  where fun is smooth and g is possibly non-smooth, using proximal gradient
  descent, also known as (F)ISTA. This method is a specific instance of
  (accelerated) projected gradient descent when the prox is a projection and
  (acclerated) gradient descent when prox is None.

  The stopping criterion is

  ||x - prox(x - grad(fun)(x, params_fun), params_prox)||_2 <= tol.

  Args:
    fun: a smooth function of the form fun(x, params_fun).
    init: initialization to use for x (pytree).
    params_fun: parameters to use for fun above (pytree).
    prox: proximity operator associated with the function g.
    params_prox: parameters to use for prox above (pytree).
    stepsize: a stepsize to use (if <= 0, use backtracking line search).
    maxiter: maximum number of proximal gradient descent iterations.
    maxls: maximum number of iterations to use in the line search.
    tol: tolerance to use.
    acceleration: whether to use acceleration (also known as FISTA) or not.
    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.
    implicit_diff: whether to use implicit differentiation or not.
      implicit_diff=False will trigger loop unrolling.
    ret_info: whether to return an OptimizeResults object containing additional
      information regarding the solution

  Returns:
    If ret_info:
      An OptimizeResults object.
    Otherwise:
      Approximate solution to the problem (same pytree structure as `init`).

  References:
    Beck, Amir, and Marc Teboulle. "A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems." SIAM imaging sciences (2009)

    Nesterov, Yu. "Gradient methods for minimizing composite functions."
    Mathematical Programming (2013).
  """
  if implicit_diff:
    # We use implicit differentiation.
    _fun = jax.custom_vjp(
        _proximal_gradient, nondiff_argnums=(0, 3, 5, 6, 7, 8, 9, 10, 11, 12))
    _fun.defvjp(_proximal_gradient_fwd, _proximal_gradient_bwd)
  else:
    # We leave differentiation to JAX.
    _fun = _proximal_gradient

  return _fun(
      fun=fun,
      init=init,
      params_fun=params_fun,
      prox=prox,
      params_prox=params_prox,
      stepsize=stepsize,
      maxiter=maxiter,
      maxls=maxls,
      tol=tol,
      acceleration=acceleration,
      verbose=verbose,
      unroll=not implicit_diff,
      ret_info=ret_info)
