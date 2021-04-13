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

"""Implicit differentiation of fixed points."""

from typing import Any
from typing import Callable
from typing import Tuple

import jax
from jax.scipy.sparse import linalg as sparse_linalg

from jaxopt.tree_util import tree_sub


def _precompute_fixed_point_vjp(vjp_fun: Callable, cotangent: Any) -> Any:
  """Pre-computes the solution of the linear system needed for the
  vector-Jacobian product of a fixed point.

  This function allows to solve the linear system only once when the fixed
  point has several parameters that need to be differentiated (e.g.,
  params_fun and params_prox in the proximal gradient fixed point).

  Args:
    vjp_fun: the vjp operator for fixed_point_fun.
    cotangent: vector to left-multiply the Jacobian with
      (same pytree structure as `sol`).
  Returns:
    The solution of A^T u = v, where A = Id - J, v = cotangent, and
    J = jacobian(fixed_point_fun, argnums=0).
  """
  def matvec(u):
    # Compute the multiplication A^T u = (u^T A)^T = (u^T (Id - J))^T.
    uJ = vjp_fun(u)[0]
    return tree_sub(u, uJ)
  return sparse_linalg.cg(matvec, cotangent)[0]


def fixed_point_vjp(fixed_point_fun: Callable,
                    params_fun: Any,
                    sol: Any,
                    cotangent: Any) -> Any:
  """Vector-Jacobian product of a fixed point function.

  The fixed point is x = fixed_point_fun(x, params_fun).

  Args:
    fixed_point_fun: the fixed point function to use.
    params_fun: parameters to use for fixed_point_fun above.
    sol: solution of the fixed point.
    cotangent: vector to left-multiply the Jacobian with
      (same pytree structure as `sol`).
  Returns:
    Vector-Jacobian product w.r.t. `params_fun` of `sol` with cotangent,
    which has the same pytree  structure as `params_fun`.
  """
  _, vjp_fun = jax.vjp(fixed_point_fun, sol, params_fun)
  u = _precompute_fixed_point_vjp(vjp_fun, cotangent)
  return vjp_fun(u)[1]


def _jvp1(f, primals, tangent):
  """JVP in the first argument of f."""
  fun = lambda x: f(x, primals[1])
  return jax.jvp(fun, (primals[0],), (tangent,))[1]


def _jvp2(f, primals, tangent):
  """JVP in the second argument of f."""
  fun = lambda y: f(primals[0], y)
  return jax.jvp(fun, (primals[1],), (tangent,))[1]


def fixed_point_jvp(fixed_point_fun: Callable,
                    params_fun: Any,
                    sol: Any,
                    tangent: Any) -> Any:
  """Jacobian-vector product using a fixed point function.

  The fixed point is x = fixed_point_fun(x, params_fun).

  Args:
    fixed_point_fun: the fixed point function to use.
    params_fun: parameters to use for fixed_point_fun above.
    sol: solution of the fixed point.
    tangent: a pytree to right-multiply the Jacobian with, with the same pytree
      structure as `params_fun`.
  Returns:
    Jacobian-vector product w.r.t. `params_fun` of `sol` with `tangent`.
    It has the same pytree structure as `sol`.
  """
  def matvec(u):
    # Compute the multiplication Au = (Id - J)u,
    # where A = jacobian(fixed_point_fun, argnums=0).
    Ju = _jvp1(fixed_point_fun, (sol, params_fun), u)
    return tree_sub(u, Ju)

  Jv = _jvp2(fixed_point_fun, (sol, params_fun), tangent)
  return sparse_linalg.cg(matvec, Jv)[0]


def gd_fixed_point_vjp(fun: Callable,
                       params_fun: Any,
                       sol: Any,
                       cotangent: Any) -> Any:
  """Vector-Jacobian product of the gradient descent fixed point.

  The fixed point is x = x - grad(fun)(x, params_fun).

  Args:
    fun: a smooth function of the form fun(x, params_fun).
    params_fun: parameters to use for fun above.
    sol: solution of the fixed point.
    cotangent: vector to left-multiply the Jacobian with
      (same pytree structure as `sol`).
  Returns:
    vjp_params_fun, which is the vector-Jacobian product of `sol` with cotangent
      and has the same pytree structure as `params_fun`.
  """
  grad_fun = jax.grad(fun)
  fixed_point_fun = lambda x, params: tree_sub(x, grad_fun(x, params))
  return fixed_point_vjp(fixed_point_fun, params_fun, sol, cotangent)


def gd_fixed_point_jvp(fun: Callable,
                       params_fun: Any,
                       sol: Any,
                       tangent: Any) -> Any:
  """Jacobian-vector product of the gradient descent fixed point.

  The fixed point is x = x - grad(fun)(x, params_fun).

  Args:
    fun: a smooth function of the form fun(x, params_fun).
    params_fun: parameters to use for fun above.
    sol: solution of the fixed point.
    tangent: a pytree to right-multiply the Jacobian with, with the same pytree
      structure as `params_fun`.
  Returns:
    Jacobian-vector product w.r.t. `params_fun` of `sol` with `tangent`.
    It has the same pytree structure as `sol`.
  """
  grad_fun = jax.grad(fun)
  fixed_point_fun = lambda x, params: tree_sub(x, grad_fun(x, params))
  return fixed_point_jvp(fixed_point_fun, params_fun, sol, tangent)


def pg_fixed_point_vjp(fun: Callable,
                       params_fun: Any,
                       prox: Callable,
                       params_prox: Any,
                       sol: Any,
                       cotangent: Any) -> Tuple[Any, Any]:
  """Vector-Jacobian product of the proximal gradient fixed point.

  The fixed point is:
    x = prox(x - grad(fun)(x, params_fun), params_prox)

  Args:
    fun: a smooth function of the form fun(x, params_fun).
    params_fun: parameters to use for fun above.
    prox: proximity operator to use.
    params_prox: parameters to use for prox above.
    sol: solution of the fixed point.
    cotangent: vector to left-multiply the Jacobian with
      (same pytree structure as `sol`).
  Returns:
    (vjp_params_fun, vjp_params_prox), which have the same pytree structure as
    `params_fun` and `params_prox`, respectively.
  """
  grad_fun = jax.grad(fun)
  fixed_point_fun = lambda x, pf, pp: prox(tree_sub(x, grad_fun(x, pf)), pp)

  _, vjp_fun = jax.vjp(fixed_point_fun, sol, params_fun, params_prox)
  u = _precompute_fixed_point_vjp(vjp_fun, cotangent)
  vjp = vjp_fun(u)
  return vjp[1], vjp[2]


def pg_fixed_point_jvp(fun: Callable,
                       params_fun: Any,
                       prox: Callable,
                       params_prox: Any,
                       sol: Any,
                       tangents: Any) -> Tuple[Any, Any]:
  """Jacobian-vector product of the proximal gradient fixed point.

  The fixed point is:
    x = prox(x - grad(fun)(x, params_fun), params_prox)

  Args:
    fun: a smooth function of the form fun(x, params_fun).
    params_fun: parameters to use for fun above.
    prox: proximity operator to use.
    params_prox: parameters to use for prox above.
    sol: solution of the fixed point.
    tangents: a tuple containing the vectors to right-multiply the Jacobian
      with, where tangents[0] has the same pytree structure as `params_fun` and
      tangents[1] has the same pytree structure as `params_prox`.
  Returns:
    (jvp_params_fun, jvp_params_prox)

    where `jvp_params_fun` and `jvp_params_prox` are the Jacobian-vector product
    of `sol` with `tangents[0]` and `tangents[1]`, respectively. Both have the
    same pytree structure as `sol`.
  """
  grad_fun = jax.grad(fun)
  fp_fun1 = lambda x, pf: prox(tree_sub(x, grad_fun(x, pf)), params_prox)
  fp_fun2 = lambda x, pp: prox(tree_sub(x, grad_fun(x, params_fun)), pp)

  jvp1 = fixed_point_jvp(fp_fun1, params_fun, sol, tangents[0])
  jvp2 = fixed_point_jvp(fp_fun2, params_prox, sol, tangents[1])
  return jvp1, jvp2


def _gd_fixed_point(solver_fun, fun):
  def solver_fun_fwd(params_fun):
    sol = solver_fun(params_fun)
    return sol, (params_fun, sol)

  def solver_fun_bwd(res, cotangent):
    params_fun, sol = res
    return (gd_fixed_point_vjp(fun=fun, sol=sol, params_fun=params_fun,
                               cotangent=cotangent),)

  wrapped_solver_fun = jax.custom_vjp(solver_fun)
  wrapped_solver_fun.defvjp(solver_fun_fwd, solver_fun_bwd)

  return wrapped_solver_fun


def gd_fixed_point(fun):
  """Decorator for adding implicit differentiation to a solver.

  Args:
    fun: a smooth function of the form fun(x, params_fun).

  Returns:
    solver function wrapped with implicit differentiation.

  Example::

    def fun(W, lam):
      logits = jnp.dot(X, W)
      return (jnp.sum(jax.vmap(loss.multiclass_logistic_loss)(y, logits)) +
              0.5 * lam * jnp.sum(W ** 2))

    @gd_fixed_point(fun)
    def solver_fun(lam):
      return your_favorite_logreg_solver(X, y, lam)

    jac_lam = jax.jacrev(solver_fun)(10.0)
  """
  def wrapper(solver_fun):
    return _gd_fixed_point(solver_fun, fun)
  return wrapper


def _pg_fixed_point(solver_fun, fun, prox):
  def solver_fun_fwd(params_fun, params_prox):
    sol = solver_fun(params_fun, params_prox)
    return sol, (params_fun, params_prox, sol)

  def solver_fun_bwd(res, cotangent):
    params_fun, params_prox, sol = res
    if prox is None:
      return (gd_fixed_point_vjp(fun=fun, sol=sol, params_fun=params_fun,
                                 cotangent=cotangent),)
    else:
      return pg_fixed_point_vjp(fun=fun, sol=sol, params_fun=params_fun,
                                prox=prox, params_prox=params_prox,
                                cotangent=cotangent)

  wrapped_solver_fun = jax.custom_vjp(solver_fun)
  wrapped_solver_fun.defvjp(solver_fun_fwd, solver_fun_bwd)

  return wrapped_solver_fun


def pg_fixed_point(fun, prox):
  """Decorator for adding implicit differentiation to a solver.

  Args:
    fun: a smooth function of the form fun(x, params_fun).
    prox: proximity operator to use.

  Return:
    solver function wrapped with implicit differentiation.

  Example::

    def fun(w, params_fun):
      y_pred = jnp.dot(X, w)
      diff = y_pred - y
      return 0.5 / (params_fun * X.shape[0]) * jnp.dot(diff, diff)

    @pg_fixed_point(fun, prox_lasso)
    def solver_fun(params_fun, params_prox):
      return your_favorite_lasso_solver(X, y, params_prox)

    jac_params_prox = jax.jacrev(solver_fun, argnums=1)(1.0, 10.0)
  """
  def wrapper(solver_fun):
    return _pg_fixed_point(solver_fun, fun, prox)
  return wrapper
