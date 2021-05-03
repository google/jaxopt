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

"""Implementation of block coordinate descent in JAX."""

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


def _make_bcd_epoch(linop, b, stepsizes, subfun_grad, params_fun, block_prox,
                    params_prox):
  """Creates a function ``bcd_epoch``, for performing one epoch of BCD, i.e.,
     for updating all blocks once."""



  def body_fun(i, tup):
    x, sub_g, Ax = tup
    x_i_old = x[i]
    g_i = linop.rmatvec_element(sub_g, i)
    if b is not None:
      g_i += b[i]
    x_i_new = block_prox(x[i] - stepsizes[i] * g_i, params_prox, stepsizes[i])
    x = jax.ops.index_update(x, i, x_i_new)
    Ax = linop.update_matvec(Ax, (x_i_new - x_i_old), i)
    sub_g = subfun_grad(Ax, params_fun)
    return x, sub_g, Ax

  def bcd_epoch(x, sub_g, Ax):
    init = (x, sub_g, Ax)
    return jax.lax.fori_loop(0, x.shape[0], body_fun, init)

  return jax.jit(bcd_epoch)


def make_solver_fun(fun: base.CompositeLinearFunction,
                    block_prox: Callable,
                    init: jnp.ndarray,
                    maxiter: int = 500,
                    tol: float = 1e-3,
                    verbose: int = 0,
                    implicit_diff: Union[bool,Callable] = True) -> Callable:
  """Creates a block coordinate descent solver function
  ``solver_fun(params_fun, params_prox)`` for solving::

    argmin_x fun(x, params_fun) + \sum_j g(x[j], params_prox),

  where ``fun`` is smooth and ``g`` is possibly non-smooth.
  Each ``x[j]`` denotes a block (a coordinate if ``x`` is a vector,
  a row if ``x`` is a matrix).

  The stopping criterion is::

    ||x - prox(x - grad(fun)(x, params_fun), params_prox)||_2 <= tol.

  where ``prox = jax.vmap(block_prox, in_axes=(0, None))``.

  Args:
    fun: a smooth function of the form ``fun(x, params_fun)``.
      It should be a base.CompositeLinearFunction object.
    block_prox: block-wise proximity operator associated with the function g,
      a function of the form ``block_prox(x[j], params_prox, scaling=1.0)``.
    init: initialization to use for ``x`` (pytree). It can be a vector
      or a matrix, and ``init.shape[0]`` determines the number of blocks.
    maxiter: maximum number of proximal gradient descent iterations.
    tol: tolerance to use.
    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.
    implicit_diff: if True, enable implicit differentiation using cg,
      if Callable, do implicit differentiation using callable as linear solver,
      if False, enable autodiff (this triggers loop unrolling),

  Returns:
    Solver function ``solver_fun(params_fun, params_prox)``.
  """

  if not isinstance(fun, base.CompositeLinearFunction):
    raise ValueError("`fun` should be a CompositeLinearFunction object.")

  prox = jax.vmap(block_prox, in_axes=(0, None))
  subfun_grad = jax.grad(fun.subfun)

  def solver_fun(params_fun=None, params_prox=None):
    stepsizes = 1.0 / fun.column_lipschitz_constants(params_fun)
    assert stepsizes.shape[0] == init.shape[0]

    bcd_epoch = _make_bcd_epoch(fun.linop, fun.b, stepsizes, subfun_grad,
                                params_fun, block_prox, params_prox)

    def cond_fun(args):
      iter_num = args[0]
      error = args[-1]
      if verbose:
        print(iter_num, error)
      return error > tol

    def body_fun(args):
      iter_num, x, sub_g, Ax, error = args
      x, sub_g, Ax = bcd_epoch(x, sub_g, Ax)
      g = fun.linop.rmatvec(sub_g)
      next_x = prox(x - g, params_prox)
      error = jnp.sqrt(jnp.sum((next_x - x) ** 2))
      return iter_num + 1, x, sub_g, Ax, error

    Ax = fun.linop.matvec(init)
    sub_g = subfun_grad(Ax, params_fun)
    # iter_num, x, sub_g, Ax, error
    args = (0, init, sub_g, Ax, jnp.inf)
    jit = False if verbose else bool(implicit_diff)
    res = loop.while_loop(cond_fun=cond_fun, body_fun=body_fun, init_val=args,
                          maxiter=maxiter, unroll=not jit, jit=jit)
    return res[1]

  if implicit_diff:
    if isinstance(implicit_diff, Callable):
      solve = implicit_diff
    else:
      solve = linear_solve.solve_normal_cg
    fixed_point_fun = idf.make_block_cd_fixed_point_fun(fun, block_prox)
    solver_fun = idf.custom_fixed_point(fixed_point_fun,
                                        unpack_params=True,
                                        solve=solve)(solver_fun)

  return solver_fun
