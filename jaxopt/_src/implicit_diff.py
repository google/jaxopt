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
from typing import Tuple

import numpy as np  # to be removed, this is for the first draft
import jax

from jaxopt._src import base
from jaxopt._src import linear_solve
from jaxopt._src.tree_util import tree_add
from jaxopt._src.tree_util import tree_mul
from jaxopt._src.tree_util import tree_scalar_mul
from jaxopt._src.tree_util import tree_sub


def root_vjp(optimality_fun: Callable,
             sol: Any,
             args: Tuple,
             cotangent: Any,
             solve: Callable = linear_solve.solve_normal_cg) -> Any:
  """Vector-Jacobian product of a root.

  The invariant is ``optimality_fun(sol, *args) == 0``.

  Args:
    optimality_fun: the optimality function to use.
    sol: solution / root (pytree).
    args: tuple containing the arguments with respect to which we wish to
      differentiate ``sol`` against.
    cotangent: vector to left-multiply the Jacobian with
      (pytree, same structure as ``sol``).
    solve: a linear solver of the form ``x = solve(matvec, b)``,
      where ``matvec(x) = Ax`` and ``Ax=b``.
  Returns:
    vjps: tuple of the same length as ``len(args)`` containing the vjps w.r.t.
      each argument. Each ``vjps[i]`` has the same pytree structure as
      ``args[i]``.
  """
  def fun_sol(sol):
    # We close over the arguments.
    return optimality_fun(sol, *args)

  _, vjp_fun_sol = jax.vjp(fun_sol, sol)

  # Compute the multiplication A^T u = (u^T A)^T.
  matvec = lambda u: vjp_fun_sol(u)[0]

  # The solution of A^T u = v, where
  # A = jacobian(optimality_fun, argnums=0)
  # v = -cotangent.
  v = tree_scalar_mul(-1, cotangent)
  u = solve(matvec, v)

  def fun_args(*args):
    # We close over the solution.
    return optimality_fun(sol, *args)

  _, vjp_fun_args = jax.vjp(fun_args, *args)

  return vjp_fun_args(u)


def sparse_root_vjp(optimality_fun: Callable,
                    make_restricted_optimality_fun: Callable,
                    sol: Any,
                    args: Tuple,
                    cotangent: Any,
                    solve: Callable = linear_solve.solve_cg) -> Any:
  """Sparse vector-Jacobian product of a root.

  The invariant is ``optimality_fun(sol, *args) == 0``.

  Args:
    optimality_fun: the optimality function to use.
    F in the paper
    make_restricted_optimality_fun: TODO XXX.
    sol: solution / root (pytree).
    args: tuple containing the arguments with respect to which we wish to
      differentiate ``sol`` against.
    cotangent: vector to left-multiply the Jacobian with
      (pytree, same structure as ``sol``).
    solve: a linear solver of the form, ``x = solve(matvec, b)``,
      where ``matvec(x) = Ax`` and ``Ax=b``.
  Returns:
    vjps: tuple of the same length as ``len(args)`` containing the vjps w.r.t.
      each argument. Each ``vjps[i]` has the same pytree structure as
      ``args[i]``.
  """
  support = sol != 0  # nonzeros coefficients of the solution
  restricted_sol = sol[support]  # solution restricted to the support

  restricted_optimality_fun = make_restricted_optimality_fun(support)

  def fun_sol(restricted_sol):
    # We close over the arguments.
    return restricted_optimality_fun(restricted_sol, *args)

  _, vjp_fun_sol = jax.vjp(fun_sol, restricted_sol)

  # Compute the multiplication A^T u = (u^T A)^T resticted to the support.
  def restricted_matvec(restricted_v):
    return vjp_fun_sol(restricted_v)[0]

  # The solution of A^T u = v, where
  # A = jacobian(optimality_fun, argnums=0)
  # v = -cotangent.
  restricted_v = tree_scalar_mul(-1, cotangent[support])
  restricted_u = solve(restricted_matvec, restricted_v)

  def fun_args(*args):
    # We close over the solution.
    return restricted_optimality_fun(restricted_sol, *args)

  _, vjp_fun_args = jax.vjp(fun_args, *args)

  return vjp_fun_args(restricted_u)

def sparse_root_vjp2(optimality_fun: Callable,
                    # filter_args: Callable,
                    sol: Any,
                    args: Tuple,
                    cotangent: Any,
                    solve: Callable = linear_solve.solve_cg) -> Any:
  """Sparse vector-Jacobian product of a root.

  The invariant is ``optimality_fun(sol, *args) == 0``.

  Args:
    optimality_fun: the optimality function to use.
    F in the paper
    make_restricted_optimality_fun: TODO XXX.
    sol: solution / root (pytree).
    args: tuple containing the arguments with respect to which we wish to
      differentiate ``sol`` against.
    cotangent: vector to left-multiply the Jacobian with
      (pytree, same structure as ``sol``).
    solve: a linear solver of the form, ``x = solve(matvec, b)``,
      where ``matvec(x) = Ax`` and ``Ax=b``.
  Returns:
    vjps: tuple of the same length as ``len(args)`` containing the vjps w.r.t.
      each argument. Each ``vjps[i]` has the same pytree structure as
      ``args[i]``.
  """
  support = sol != 0  # nonzeros coefficients of the solution
  restricted_sol = sol[support]  # solution restricted to the support

  X, y, lam = args
  new_args = X[:, support], y, lam

  def fun_sol(restricted_sol):
    # We close over the arguments.
    return optimality_fun(restricted_sol, *new_args)

  _, vjp_fun_sol = jax.vjp(fun_sol, restricted_sol)

  # Compute the multiplication A^T u = (u^T A)^T resticted to the support.
  def restricted_matvec(restricted_v):
    return vjp_fun_sol(restricted_v)[0]

  # The solution of A^T u = v, where
  # A = jacobian(optimality_fun, argnums=0)
  # v = -cotangent.
  restricted_v = tree_scalar_mul(-1, cotangent[support])
  restricted_u = solve(restricted_matvec, restricted_v)

  def fun_args(*args):
    # We close over the solution.
    X, y, lam = args
    new_args = X[:, support], y, lam
    return optimality_fun(restricted_sol, *new_args)

  _, vjp_fun_args = jax.vjp(fun_args, *args)
  # _, vjp_fun_args = jax.vjp(fun_args, *new_args)

  return vjp_fun_args(restricted_u)


def _jvp_sol(optimality_fun, sol, args, tangent):
  """JVP in the first argument of optimality_fun."""
  # We close over the arguments.
  fun = lambda x: optimality_fun(x, *args)
  return jax.jvp(fun, (sol,), (tangent,))[1]


def _jvp_args(optimality_fun, sol, args, tangents):
  """JVP in the second argument of optimality_fun."""
  # We close over the solution.
  fun = lambda *y: optimality_fun(sol, *y)
  return jax.jvp(fun, args, tangents)[1]


def root_jvp(optimality_fun: Callable,
             sol: Any,
             args: Tuple,
             tangents: Tuple,
             solve:Callable = linear_solve.solve_normal_cg) -> Any:
  """Jacobian-vector product of a root.

  The invariant is ``sol = optimality_fun(sol, *args) == 0``.

  Args:
    optimality_fun: the optimality function to use.
    sol: solution / root (pytree).
    args: tuple containing the arguments with respect to which to differentiate.
    tangents: a tuple of the same size as ``len(args)``. Each ``tangents[i]``
      has the same pytree structure as ``args[i]``.
    solve: a linear solver of the form ``solve(matvec, b)``.
  Returns:
    jvp: a pytree with the same structure as ``sol``.
  """
  if len(args) != len(tangents):
    raise ValueError("args and tangents should be tuples of the same length.")

  # Product with A = jacobian(fun, argnums=0).
  matvec = lambda u: _jvp_sol(optimality_fun, sol, args, u)

  v = tree_scalar_mul(-1, tangents)
  Jv = _jvp_args(optimality_fun, sol, args, v)
  return solve(matvec, Jv)


def _custom_root(solver_fun, optimality_fun, solve, has_aux):
  def solver_fun_fwd(init_params, *args):
    res = solver_fun(init_params, *args)
    return res, (res, args)

  def solver_fun_bwd(tup, cotangent):
    res, args = tup

    # solver_fun can return auxiliary data if has_aux = True.
    if has_aux:
      cotangent = cotangent[0]
      sol = res[0]
    else:
      sol = res

    # Compute VJPs w.r.t. args.
    vjps = root_vjp(optimality_fun=optimality_fun, sol=sol, args=args,
                    cotangent=cotangent, solve=solve)
    # For init_params, we return None.
    return (None,) + vjps

  wrapped_solver_fun = jax.custom_vjp(solver_fun)
  wrapped_solver_fun.defvjp(solver_fun_fwd, solver_fun_bwd)

  return wrapped_solver_fun


def _sparse_custom_root(
  solver_fun, optimality_fun, make_restricted_optimality_fun, solve, has_aux):
  def solver_fun_fwd(init_params, *args):
    res = solver_fun(init_params, *args)
    return res, (res, args)

  def solver_fun_bwd(tup, cotangent):
    res, args = tup

    # solver_fun can return auxiliary data if has_aux = True.
    if has_aux:
      cotangent = cotangent[0]
      sol = res[0]
    else:
      sol = res

    # Compute VJPs w.r.t. args.
    vjps = sparse_root_vjp(
      optimality_fun=optimality_fun,
      make_restricted_optimality_fun=make_restricted_optimality_fun,
      sol=sol, args=args, cotangent=cotangent, solve=solve)
    # For init_params, we return None.
    return (None,) + vjps

  wrapped_solver_fun = jax.custom_vjp(solver_fun)
  wrapped_solver_fun.defvjp(solver_fun_fwd, solver_fun_bwd)

  return wrapped_solver_fun


def _sparse_custom_root2(
  solver_fun, optimality_fun, solve, has_aux):
  def solver_fun_fwd(init_params, *args):
    res = solver_fun(init_params, *args)
    return res, (res, args)

  def solver_fun_bwd(tup, cotangent):
    res, args = tup

    # solver_fun can return auxiliary data if has_aux = True.
    if has_aux:
      cotangent = cotangent[0]
      sol = res[0]
    else:
      sol = res

    # Compute VJPs w.r.t. args.
    vjps = sparse_root_vjp2(
      optimality_fun=optimality_fun,
      sol=sol, args=args, cotangent=cotangent, solve=solve)
    # For init_params, we return None.
    return (None,) + vjps

  wrapped_solver_fun = jax.custom_vjp(solver_fun)
  wrapped_solver_fun.defvjp(solver_fun_fwd, solver_fun_bwd)

  return wrapped_solver_fun


def custom_root(optimality_fun: Callable,
                has_aux: bool = False,
                solve: Callable = linear_solve.solve_normal_cg):
  """Decorator for adding implicit differentiation to a root solver.

  Args:
    optimality_fun: an equation function, ``optimality_fun(params, *args)``.
      The invariant is ``optimality_fun(sol, *args) == 0`` at the
      solution / root ``sol``.
    has_aux: whether the decorated solver function returns auxiliary data.
    solve: a linear solver of the form ``solve(matvec, b)``.

  Returns:
    A solver function decorator, i.e.,
      ``custom_root(optimality_fun)(solver_fun)``.
  """
  def wrapper(solver_fun):
    return _custom_root(solver_fun, optimality_fun, solve, has_aux)
  return wrapper


def sparse_custom_root(optimality_fun: Callable,
                       make_restricted_optimality_fun: Callable,
                       has_aux: bool = False,
                       solve: Callable = linear_solve.solve_normal_cg):
  """Decorator for adding implicit differentiation to a root solver.

  Args:
    optimality_fun: an equation function, ``optimality_fun(params, *args)`.
      The invariant is ``optimality_fun(sol, *args) == 0`` at the
      solution / root ``sol``.
    has_aux: whether the decorated solver function returns auxiliary data.
    solve: a linear solver of the form, ``solve(matvec, b)``.

  Returns:
    A solver function decorator, i.e.,
      ``custom_root(optimality_fun)(solver_fun)``.
  """
  def wrapper(solver_fun):
    return _sparse_custom_root(
      solver_fun, optimality_fun, make_restricted_optimality_fun, solve,
      has_aux)
  return wrapper


def sparse_custom_root2(
    optimality_fun: Callable, has_aux: bool = False,
    solve: Callable = linear_solve.solve_normal_cg):
  """Decorator for adding implicit differentiation to a root solver.

  Args:
    optimality_fun: an equation function, ``optimality_fun(params, *args)`.
      The invariant is ``optimality_fun(sol, *args) == 0`` at the
      solution / root ``sol``.
    has_aux: whether the decorated solver function returns auxiliary data.
    solve: a linear solver of the form, ``solve(matvec, b)``.

  Returns:
    A solver function decorator, i.e.,
      ``custom_root(optimality_fun)(solver_fun)``.
  """
  def wrapper(solver_fun):
    return _sparse_custom_root2(
      solver_fun, optimality_fun, solve, has_aux)

  return wrapper


def custom_fixed_point(fixed_point_fun: Callable,
                       has_aux: bool = False,
                       solve: Callable = linear_solve.solve_normal_cg):
  """Decorator for adding implicit differentiation to a fixed point solver.

  Args:
    fixed_point_fun: a function, ``fixed_point_fun(params, *args)``.
      The invariant is ``fixed_point_fun(sol, *args) == sol`` at the
      solution ``sol``.
    has_aux: whether the decorated solver function returns auxiliary data.
    solve: a linear solver of the form ``solve(matvec, b)``.

  Returns:
    A solver function decorator, i.e.,
      ``custom_fixed_point(fixed_point_fun)(solver_fun)``.
  """
  def optimality_fun(params, *args):
    return tree_sub(fixed_point_fun(params, *args), params)

  return custom_root(optimality_fun=optimality_fun,
                     has_aux=has_aux,
                     solve=solve)


def make_kkt_optimality_fun(obj_fun, eq_fun, ineq_fun=None):
  """Makes the optimality function for KKT conditions.

  Args:
    obj_fun: objective function ``obj_fun(primal_var, params_obj)``.
    eq_fun: equality constraint function, so that
      ``eq_fun(primal_var, params_eq) == 0`` is imposed.
    ineq_fun: inequality constraint function, so that
      ``ineq_fun(primal_var, params_ineq) <= 0`` is imposed (optional).
  Returns:
    optimality_fun(params, params_obj, params_eq, params_ineq) where
      x = (primal_var, eq_dual_var, ineq_dual_var)

    If ``ineq_fun`` is None, ``ineq_dual_var`` and ``params_ineq`` are
    ignored (i.e., they can be set to ``None``).
  """
  grad_fun = jax.grad(obj_fun)

  # We only consider the stationarity, primal_feasability and comp_slackness
  # conditions, as primal and dual feasibility conditions can be ignored
  # almost everywhere.
  def optimality_fun(params, params_obj, params_eq, params_ineq):
    primal_var, eq_dual_var, ineq_dual_var = params

    # Same pytree structure as the primal variable.
    _, eq_vjp_fun = jax.vjp(eq_fun, primal_var, params_eq)
    stationarity = tree_add(grad_fun(primal_var, params_obj),
                            eq_vjp_fun(eq_dual_var)[0])

    # Size: number of equality constraints.
    primal_feasability = eq_fun(primal_var, params_eq)

    if params_ineq is not None:
      _, ineq_vjp_fun = jax.vjp(ineq_fun, primal_var, params_ineq)

      stationarity = tree_add(stationarity, ineq_vjp_fun(ineq_dual_var)[0])

      # Size: number of inequality constraints.
      comp_slackness = tree_mul(ineq_fun(primal_var, params_ineq),
                                ineq_dual_var)

      return base.KKTSolution(stationarity, primal_feasability, comp_slackness)
    else:
      return base.KKTSolution(stationarity, primal_feasability)

  return optimality_fun
