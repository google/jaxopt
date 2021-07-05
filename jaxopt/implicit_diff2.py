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

"""Implicit differentiation of roots and fixed points."""


from typing import Any
from typing import Callable

import jax
import jax.numpy as jnp

from jaxopt import linear_solve
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub


def root_vjp(optimality_fun: Callable,
             sol: Any,
             hyperparams: Any,
             cotangent: Any,
             solve: Callable = linear_solve.solve_normal_cg,
             *args,
             **kw) -> Any:
  """Vector-Jacobian product of a root.

  The invariant is ``optimality_fun(sol, hyperparams, *args, **kw) == 0``.

  Args:
    optimality_fun: the optimality function to use.
    sol: solution / root (pytree).
    hyperparams: hyper-parameters to use for ``optimality_fun`` above (pytree).
    cotangent: vector to left-multiply the Jacobian with
      (pytree, same structure as ``sol``).
    solve: a linear solver of the form, ``x = solve(matvec, b)``,
      where ``matvec(x) = Ax`` and ``Ax=b``.
    *args, **kw: additional arguments to be passed to ``optimality_fun``.
  Returns:
    Vector-Jacobian product w.r.t. ``hyperparams` of `sol` with cotangent.
    It has the same pytree  structure as `hyperparams`.
  """
  # We close over the extra *args and **kw as we only differentiate
  # w.r.t. hyperparams.
  fun = lambda x, hp: optimality_fun(x, hp, *args, **kw)
  _, vjp_fun = jax.vjp(fun, sol, hyperparams)

  # Compute the multiplication A^T u = (u^T A)^T.
  matvec = lambda u: vjp_fun(u)[0]

  # The solution of A^T u = v, where
  # A = jacobian(fun, argnums=0)
  # v = -cotangent.
  v = tree_scalar_mul(-1, cotangent)
  u = solve(matvec, v)

  return vjp_fun(u)[1]


def _jvp1(f, primals, tangent):
  """JVP in the first argument of f."""
  fun = lambda x: f(x, primals[1])
  return jax.jvp(fun, (primals[0],), (tangent,))[1]


def _jvp2(f, primals, tangent):
  """JVP in the second argument of f."""
  fun = lambda y: f(primals[0], y)
  return jax.jvp(fun, (primals[1],), (tangent,))[1]


def root_jvp(optimality_fun: Callable,
             sol: Any,
             hyperparams: Any,
             tangent: Any,
             solve:Callable = linear_solve.solve_normal_cg,
             *args,
             **kw) -> Any:
  """Jacobian-vector product of a root.

  The invariant is ``sol = fun(sol, hyperparams, *args, **kw) == 0``.

  Args:
    optimality_fun: the optimality function to use.
    sol: solution / root (pytree).
    hyperparams: hyper-parameters to use for ``optimality_fun`` above (pytree).
    tangent: a pytree to right-multiply the Jacobian with, with the same pytree
      structure as ``hyperparams``.
    solve: a linear solver of the form, ``solve(matvec, b)``.
    *args, **kw: additional arguments to be passed to ``optimality_fun``.
  Returns:
    Jacobian-vector product w.r.t. ``hyperparams`` of ``sol`` with ``tangent``.
    It has the same pytree structure as ``sol``.
  """
  # We close over the extra *args and **kw as we only differentiate
  # w.r.t. hyperparams.
  fun = lambda x, hp: optimality_fun(x, hp, *args, **kw)

  # Product with A = jacobian(fun, argnums=0).
  matvec = lambda u: _jvp1(fun, (sol, hyperparams), u)

  v = tree_scalar_mul(-1, tangent)
  Jv = _jvp2(fun, (sol, hyperparams), v)
  return solve(matvec, Jv)


def _custom_root(solver_fun, optimality_fun, solve, has_aux):
  def solver_fun_fwd(*args):
    res = solver_fun(*args)
    return res, (res, args)

  def solver_fun_bwd(tup, cotangent):
    res, args = tup

    # solver_fun can return auxiliary data if has_aux = True.

    if has_aux:
      cotangent = cotangent[0]
      sol = res[0]
    else:
      sol = res

    # solver_fun can have 1 or 3 arguments.

    if len(args) == 1:  # solver_fun(hyperparams)
      vjp_hparams = root_vjp(optimality_fun=optimality_fun, solve=solve,
                             sol=sol, hyperparams=args[0], cotangent=cotangent)
      return (vjp_hparams,)

    elif len(args) == 3:  # solver_fun(init_params, hyperparams, data)
      vjp_hparams = root_vjp(optimality_fun=optimality_fun, solve=solve,
                             sol=sol, hyperparams=args[1], data=args[2],
                             cotangent=cotangent)
      return (None,) + (vjp_hparams,) + (None,)

    else:
      raise ValueError("Invalid number of arguments in solver function.")

  wrapped_solver_fun = jax.custom_vjp(solver_fun)
  wrapped_solver_fun.defvjp(solver_fun_fwd, solver_fun_bwd)

  return wrapped_solver_fun


def custom_root(optimality_fun: Callable,
                has_aux: bool = False,
                solve: Callable = linear_solve.solve_normal_cg):
  """Decorator for adding implicit differentiation to a root solver.

  Args:
    optimality_fun: an equation function, ``optimality_fun(x, params)`.
      The invariant is ``optimality_fun(sol, hyperparams, data) == 0`` at the
        solution / root ``sol``.
    has_aux: whether the decorated solver function returns auxiliary data.
    solve: a linear solver of the form, ``solve(matvec, b)``.

  Returns:
    A solver function decorator, i.e.,
      ``custom_root(optimality_fun)(solver_fun)``.
  """
  def wrapper(solver_fun):
    return _custom_root(solver_fun, optimality_fun, solve, has_aux)
  return wrapper
