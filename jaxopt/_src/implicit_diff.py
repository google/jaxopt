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

import inspect
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

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
    tuple of the same length as ``len(args)`` containing the vjps w.r.t.
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

  The invariant is ``optimality_fun(sol, *args) == 0``.

  Args:
    optimality_fun: the optimality function to use.
    sol: solution / root (pytree).
    args: tuple containing the arguments with respect to which to differentiate.
    tangents: a tuple of the same size as ``len(args)``. Each ``tangents[i]``
      has the same pytree structure as ``args[i]``.
    solve: a linear solver of the form ``solve(matvec, b)``.
  Returns:
    a pytree with the same structure as ``sol``.
  """
  if len(args) != len(tangents):
    raise ValueError("args and tangents should be tuples of the same length.")

  # Product with A = jacobian(fun, argnums=0).
  matvec = lambda u: _jvp_sol(optimality_fun, sol, args, u)

  v = tree_scalar_mul(-1, tangents)
  Jv = _jvp_args(optimality_fun, sol, args, v)
  return solve(matvec, Jv)


def _extract_kwargs(kwarg_keys, flat_args):
  n = len(flat_args) - len(kwarg_keys)
  args, kwarg_vals = flat_args[:n], flat_args[n:]
  kwargs = dict(zip(kwarg_keys, kwarg_vals))
  return args, kwargs


def _signature_bind(signature, *args, **kwargs):
  ba = signature.bind(*args, **kwargs)
  ba.apply_defaults()
  return ba.args, ba.kwargs


def _signature_bind_and_match(signature, *args, **kwargs):
  # We want to bind *args and **kwargs based on the provided
  # signature, but also to associate the resulting positional
  # arguments back. To achieve this, we lift arguments to a triple:
  #
  #   (was_kwarg, ref, value)
  #
  # where ref is an index position (int) if the original argument was
  # from *args and a dictionary key if the original argument was from
  # **kwargs. After binding to the inspected signature, we use the
  # tags to associate the resolved positional arguments back to their
  # arg and kwarg source.

  args = [(False, i, v) for i, v in enumerate(args)]
  kwargs = {k: (True, k, v) for (k, v) in kwargs.items()}
  ba = signature.bind(*args, **kwargs)

  mapping = [(was_kwarg, ref) for was_kwarg, ref, _ in ba.args]

  def map_back(out_args):
    src_args = [None] * len(args)
    src_kwargs = {}
    for (was_kwarg, ref), out_arg in zip(mapping, out_args):
      if was_kwarg:
        src_kwargs[ref] = out_arg
      else:
        src_args[ref] = out_arg
    return src_args, src_kwargs

  out_args = tuple(v for _, _, v in ba.args)
  out_kwargs = {k: v for k, (_, _, v) in ba.kwargs.items()}
  return out_args, out_kwargs, map_back


def _custom_root(solver_fun, optimality_fun, solve, has_aux,
                 reference_signature=None):
  # When caling through `jax.custom_vjp`, jax attempts to resolve all
  # arguments passed by keyword to positions (this is in order to
  # match against a `nondiff_argnums` parameter that we do not use
  # here). It does so by resolving them according to the custom_jvp'ed
  # function's signature. It disallows functions defined with a
  # catch-all `**kwargs` expression, since their signature cannot
  # always resolve all keyword arguments to positions.
  #
  # We can loosen the constraint on the signature of `solver_fun` so
  # long as we resolve keywords to positions ourselves. We can do so
  # just in time, by flattening the `kwargs` dict (respecting its
  # iteration order) and supplying `custom_vjp` with a
  # positional-argument-only function. We then explicitly coordinate
  # flattening and unflattening around the `custom_vjp` boundary.
  #
  # Once we make it past the `custom_vjp` boundary, we do some more
  # work to align arguments with the reference signature (which is, by
  # default, the signature of `optimality_fun`).

  solver_fun_signature = inspect.signature(solver_fun)

  if reference_signature is None:
    reference_signature = inspect.signature(optimality_fun)

  elif not isinstance(reference_signature, inspect.Signature):
    # If is a CompositeLinearFunction, accesses subfun.
    # Otherwise, assumes a Callable.
    fun = getattr(reference_signature, "subfun", reference_signature)
    reference_signature = inspect.signature(fun)

  def make_custom_vjp_solver_fun(solver_fun, kwarg_keys):
    @jax.custom_vjp
    def solver_fun_flat(*flat_args):
      args, kwargs = _extract_kwargs(kwarg_keys, flat_args)
      return solver_fun(*args, **kwargs)

    def solver_fun_fwd(*flat_args):
      res = solver_fun_flat(*flat_args)
      return res, (res, flat_args)

    def solver_fun_bwd(tup, cotangent):
      res, flat_args = tup
      args, kwargs = _extract_kwargs(kwarg_keys, flat_args)

      # solver_fun can return auxiliary data if has_aux = True.
      if has_aux:
        cotangent = cotangent[0]
        sol = res[0]
      else:
        sol = res

      ba_args, ba_kwargs, map_back = _signature_bind_and_match(
          reference_signature, *args, **kwargs)
      if ba_kwargs:
        raise TypeError(
            "keyword arguments to solver_fun could not be resolved to "
            "positional arguments based on the signature "
            f"{reference_signature}. This can happen under custom_root if "
            "optimality_fun takes catch-all **kwargs, or under "
            "custom_fixed_point if fixed_point_fun takes catch-all **kwargs, "
            "both of which are currently unsupported.")

      # Compute VJPs w.r.t. args.
      vjps = root_vjp(optimality_fun=optimality_fun, sol=sol,
                      args=ba_args[1:], cotangent=cotangent, solve=solve)
      # Prepend None as the vjp for init_params.
      vjps = (None,) + vjps

      arg_vjps, kws_vjps = map_back(vjps)
      ordered_vjps = tuple(arg_vjps) + tuple(kws_vjps[k] for k in kwargs.keys())
      return ordered_vjps

    solver_fun_flat.defvjp(solver_fun_fwd, solver_fun_bwd)
    return solver_fun_flat

  def wrapped_solver_fun(*args, **kwargs):
    args, kwargs = _signature_bind(solver_fun_signature, *args, **kwargs)
    keys, vals = list(kwargs.keys()), list(kwargs.values())
    return make_custom_vjp_solver_fun(solver_fun, keys)(*args, *vals)

  return wrapped_solver_fun


def custom_root(optimality_fun: Callable,
                has_aux: bool = False,
                solve: Callable = linear_solve.solve_normal_cg,
                reference_signature: Optional[Callable] = None):
  """Decorator for adding implicit differentiation to a root solver.

  Args:
    optimality_fun: an equation function, ``optimality_fun(params, *args)``.
      The invariant is ``optimality_fun(sol, *args) == 0`` at the
      solution / root ``sol``.
    has_aux: whether the decorated solver function returns auxiliary data.
    solve: a linear solver of the form ``solve(matvec, b)``.
    reference_signature: optional function signature
      (i.e. arguments and keyword arguments), with which the
      solver and optimality functions are expected to agree. Defaults
      to ``optimality_fun``. It is required that solver and optimality
      functions share the same input signature, but both might be
      defined in such a way that the signature correspondence is
      ambiguous (e.g. if both accept catch-all ``**kwargs``). To
      satisfy custom_root's requirement, any function with an
      unambiguous signature can be provided here.

  Returns:
    A solver function decorator, i.e.,
    ``custom_root(optimality_fun)(solver_fun)``.
  """
  if solve is None:
    solve = linear_solve.solve_normal_cg

  def wrapper(solver_fun):
    return _custom_root(solver_fun, optimality_fun, solve, has_aux,
                        reference_signature)

  return wrapper


def custom_fixed_point(fixed_point_fun: Callable,
                       has_aux: bool = False,
                       solve: Callable = linear_solve.solve_normal_cg,
                       reference_signature: Optional[Callable] = None):
  """Decorator for adding implicit differentiation to a fixed point solver.

  Args:
    fixed_point_fun: a function, ``fixed_point_fun(params, *args)``.
      The invariant is ``fixed_point_fun(sol, *args) == sol`` at the
      solution ``sol``.
    has_aux: whether the decorated solver function returns auxiliary data.
    solve: a linear solver of the form ``solve(matvec, b)``.
    reference_signature: optional function whose signature
      (i.e. arguments and keyword arguments) is one with which the
      solver and fixed-point functions are expected to agree. Defaults
      to ``fixed_point_fun``. It is required that solver and
      fixed-point functions share the same input signature, but both
      might be defined in such a way that the signature correspondence
      is ambiguous (e.g. if both accept catch-all ``**kwargs``). To
      satisfy custom_fixed_points's requirement, any function with an
      unambiguous signature can be provided here.

  Returns:
    A solver function decorator, i.e.,
    ``custom_fixed_point(fixed_point_fun)(solver_fun)``.
  """
  def optimality_fun(params, *args):
    return tree_sub(fixed_point_fun(params, *args), params)

  # carry over fixed_point_fun's signature
  optimality_fun.__wrapped__ = fixed_point_fun

  return custom_root(optimality_fun=optimality_fun,
                     has_aux=has_aux,
                     solve=solve,
                     reference_signature=reference_signature)


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

    stationarity = grad_fun(primal_var, params_obj)

    if eq_dual_var is not None:
      _, eq_vjp_fun = jax.vjp(eq_fun, primal_var, params_eq)
      stationarity = tree_add(stationarity, eq_vjp_fun(eq_dual_var)[0])
      # Size: number of equality constraints.
      primal_feasability = eq_fun(primal_var, params_eq)
    else:
      primal_feasability = None

    if ineq_dual_var is not None:
      _, ineq_vjp_fun = jax.vjp(ineq_fun, primal_var, params_ineq)
      stationarity = tree_add(stationarity, ineq_vjp_fun(ineq_dual_var)[0])
      # Size: number of inequality constraints.
      comp_slackness = tree_mul(ineq_fun(primal_var, params_ineq),
                                ineq_dual_var)
    else:
      comp_slackness = None

    return base.KKTSolution(stationarity, primal_feasability, comp_slackness)

  return optimality_fun
