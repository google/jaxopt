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


def root_vjp(fun: Callable,
             sol: Any,
             params: Any,
             cotangent: Any,
             solve: Callable = linear_solve.solve_normal_cg) -> Any:
  """Vector-Jacobian product of a root.

  The invariant is ``fun(sol, params) == 0``.

  Args:
    fun: the equation function to use.
    sol: solution / root (pytree).
    params: parameters to use for ``fun`` above (pytree).
    cotangent: vector to left-multiply the Jacobian with
      (pytree, same structure as ``sol``).
    solve: a linear solver of the form, ``x = solve(matvec, b)``,
      where ``matvec(x) = Ax`` and ``Ax=b``.
  Returns:
    Vector-Jacobian product w.r.t. ``params` of `sol` with cotangent.
    It has the same pytree  structure as `params`.
  """
  _, vjp_fun = jax.vjp(fun, sol, params)

  # Compute the multiplication A^T u = (u^T A)^T.
  matvec = lambda u: vjp_fun(u)[0]

  # The solution of A^T u = v, where
  # A = jacobian(fun, argnums=0)
  # v = -cotangent.
  v = tree_scalar_mul(-1, cotangent)
  u = solve(matvec, v)

  return vjp_fun(u)[1]


def fixed_point_vjp(fixed_point_fun: Callable,
                    sol: Any,
                    params: Any,
                    cotangent: Any,
                    solve: Callable = linear_solve.solve_normal_cg) -> Any:
  """Vector-Jacobian product of a fixed point.

  The fixed point is `sol = fixed_point_fun(sol, params)`.

  Args:
    fixed_point_fun: the fixed point function to use.
    sol: solution of the fixed point (pytree).
    params: parameters to use for `fixed_point_fun` above (pytree).
    cotangent: vector to left-multiply the Jacobian with
      (pytree, same structure as `sol`).
    solve: a linear solver of the form, ``x = solve(matvec, b)``,
      where ``matvec(x) = Ax`` and ``Ax=b``.
  Returns:
    Vector-Jacobian product w.r.t. `params` of `sol` with cotangent.
    It has the same pytree  structure as `params`.
  """
  fun = lambda x, p: tree_sub(fixed_point_fun(x, p), x)
  return root_vjp(fun, sol, params, cotangent, solve)


def _jvp1(f, primals, tangent):
  """JVP in the first argument of f."""
  fun = lambda x: f(x, primals[1])
  return jax.jvp(fun, (primals[0],), (tangent,))[1]


def _jvp2(f, primals, tangent):
  """JVP in the second argument of f."""
  fun = lambda y: f(primals[0], y)
  return jax.jvp(fun, (primals[1],), (tangent,))[1]


def root_jvp(fun: Callable,
             sol: Any,
             params: Any,
             tangent: Any,
             solve:Callable = linear_solve.solve_normal_cg) -> Any:
  """Jacobian-vector product of a root.

  The invariant is `sol = fun(sol, params) == 0`.

  Args:
    fun: the equation function to use.
    sol: solution / root (pytree).
    params: parameters to use for ``fun`` above (pytree).
    tangent: a pytree to right-multiply the Jacobian with, with the same pytree
      structure as ``params``.
    solve: a linear solver of the form, ``solve(matvec, b)``.
  Returns:
    Jacobian-vector product w.r.t. ``params`` of ``sol`` with ``tangent``.
    It has the same pytree structure as ``sol``.
  """
  # Product with A = jacobian(fun, argnums=0).
  matvec = lambda u: _jvp1(fun, (sol, params), u)

  v = tree_scalar_mul(-1, tangent)
  Jv = _jvp2(fun, (sol, params), v)
  return solve(matvec, Jv)


def fixed_point_jvp(fixed_point_fun: Callable,
                    sol: Any,
                    params: Any,
                    tangent: Any,
                    solve:Callable = linear_solve.solve_normal_cg) -> Any:
  """Jacobian-vector product of a fixed point.

  The fixed point is `sol = fixed_point_fun(sol, params)`.

  Args:
    fixed_point_fun: the fixed point function to use.
    sol: solution of the fixed point (pytree).
    params: parameters to use for `fixed_point_fun` above (pytree).
    tangent: a pytree to right-multiply the Jacobian with, with the same pytree
      structure as `params`.
    solve: a linear solver of the form, ``solve(matvec, b)``.
  Returns:
    Jacobian-vector product w.r.t. `params` of `sol` with `tangent`.
    It has the same pytree structure as `sol`.
  """
  fun = lambda x, p: tree_sub(fixed_point_fun(x, p), x)
  return root_jvp(fun, sol, params, tangent, solve)


def make_gradient_descent_fixed_point_fun(fun):
  """Makes a gradient descent fixed point function.

  The fixed point function is `sol = fixed_point_fun(sol, params_fun)`,
    where `sol = sol - grad(fun)(sol, params_fun)`.

  Args:
    fun: an objective function of the form `fun(x, params_fun)`.
  Returns:
    fixed_point_fun
  """
  grad_fun = jax.grad(fun)
  return lambda x, params: tree_sub(x, grad_fun(x, params))


def make_proximal_gradient_fixed_point_fun(fun, prox):
  """Makes a proximal gradient fixed point function.

  The fixed point function is `sol = fixed_point_fun(sol, params)`, where

      `sol = prox(sol - grad(fun)(sol, params_fun), params_prox)` and

      `params = (params_fun, params_prox)`.

  Args:
    fun: a smooth function of the form `fun(x, params_fun)`.
    prox: proximity operator of the form `prox(x, params_prox, scaling=1.0)`.
  Returns:
    fixed_point_fun
  """
  grad_fun = jax.grad(fun)
  def fixed_point_fun(sol, params):
    params_fun, params_prox = params
    return prox(tree_sub(sol, grad_fun(sol, params_fun)), params_prox)
  return fixed_point_fun


def make_block_cd_fixed_point_fun(fun, block_prox):
  """Makes a block coordinate descent fixed point function.

  The fixed point function is `sol = fixed_point_fun(sol, params)`, where::

    params = (params_fun, params_prox)
    prox = jax.vmap(block_prox, in_axes=(0, None))
    sol = prox(sol - grad(fun)(sol, params_fun), params_prox)

  Args:
    fun: a smooth function of the form `fun(x, params_fun)`.
    block_prox: block-wise proximity operator of the form
      `block_prox(x, params_prox, scaling=1.0)`.
  Returns:
    fixed_point_fun
  """
  grad_fun = jax.grad(fun)
  prox = jax.vmap(block_prox, in_axes=(0, None))

  def fixed_point_fun(sol, params):
    params_fun, params_prox = params
    grad_step = sol - grad_fun(sol, params_fun)
    return prox(grad_step, params_prox)
  return fixed_point_fun


def make_mirror_descent_fixed_point_fun(fun, projection, mapping_fun):
  """Makes a gradient descent fixed point function.

  The fixed point function is `sol = fixed_point_fun(sol, params)`, where::

    x_hat = mapping_fun(sol)
    y = x_hat - grad(fun)(sol, params_fun)
    sol = projection(y, params_proj)

  and `params = (params_fun, params_proj)`.

  Typically, `mapping_fun = grad(gen_fun)`, where `gen_fun` is the generating
  function of the Bregman divergence.

  Args:
    fun: a smooth function of the form `fun(x, params_fun)`.
    projection: projection operator of the form
      `projection(x, params_proj)`.
  Returns:
    fixed_point_fun
  """
  grad_fun = jax.grad(fun)

  def fixed_point_fun(x, params):
    params_fun, params_proj = params
    x_hat = mapping_fun(x)
    y = tree_sub(x_hat, grad_fun(x, params_fun))
    return projection(y, params_proj)

  return fixed_point_fun


def _custom_root(solver_fun, fun, unpack_params, solve):
  if unpack_params:
    def solver_fun_fwd(*params):
      sol = solver_fun(*params)
      return sol, (params, sol)
  else:
    def solver_fun_fwd(params):
      sol = solver_fun(params)
      return sol, (params, sol)

  def solver_fun_bwd(res, cotangent):
    params, sol = res
    vjp = root_vjp(fun=fun, solve=solve,sol=sol, params=params,
                   cotangent=cotangent)
    if unpack_params:
      return vjp
    else:
      return (vjp,)

  wrapped_solver_fun = jax.custom_vjp(solver_fun)
  wrapped_solver_fun.defvjp(solver_fun_fwd, solver_fun_bwd)

  return wrapped_solver_fun


def custom_root(fun: Callable,
                unpack_params: bool = False,
                solve: Callable = linear_solve.solve_normal_cg):
  """Decorator for adding implicit differentiation to a root solver.

  Args:
    fun: an equation function, ``fun(x, params)`.
      The invariant is ``fun(sol, params) == 0`` at the solution / root ``sol``.
    unpack_params: if True, the signature of the solver function must be
      ``solver_fun(*params)`` instead of ``solver_fun(params)``.
    solve: a linear solver of the form, ``solve(matvec, b)``.

  Returns:
    A solver function decorator, i.e.,
      ``custom_root(fun, unpack_params)(solver_fun)``.
  """
  def wrapper(solver_fun):
    return _custom_root(solver_fun, fun, unpack_params, solve)
  return wrapper


def custom_fixed_point(fixed_point_fun: Callable,
                       unpack_params: bool = False,
                       solve: Callable = linear_solve.solve_normal_cg):
  """Decorator for adding implicit differentiation to a fixed point solver.

  Args:
    fixed_point_fun: a fixed point function, `fixed_point_fun(x, params)`.
      The invariant is `sol == fixed_point_fun(sol, params)` at the fixed point
      `sol`.
    unpack_params: if True, the signature of the solver function must be
      ``solver_fun(*params)`` instead of ``solver_fun(params)``.
    solve: a linear solver of the form, ``solve(matvec, b)``.

  Returns:
    A solver function decorator, i.e.,
      ``custom_fixed_point(fixed_point_fun, unpack_params)(solver_fun)``.

  Example::

    from jaxopt.implicit_diff import custom_fixed_point
    from jaxopt.implicit_diff import make_gradient_descent_fixed_point_fun
    from jaxopt import loss

    from sklearn import datasets
    from sklearn import linear_model

    X, y = datasets.make_classification(n_samples=50, n_features=10,
                                        n_informative=5, n_classes=3,
                                        random_state=0)

    def fun(W, lam):
      # Objective function solved by the logistic regression solver.
      logits = jnp.dot(X, W)
      return (jnp.sum(jax.vmap(loss.multiclass_logistic_loss)(y, logits)) +
              0.5 * lam * jnp.sum(W ** 2))

    fixed_point_fun = make_gradient_descent_fixed_point_fun(fun)

    @custom_fixed_point(fixed_point_fun)
    def solver_fun(lam):
      logreg = linear_model.LogisticRegression(fit_intercept=False,
                                               C=1. / lam,
                                               multi_class="multinomial")
      return logreg.fit(X, y).coef_.T

    # Jacobian of solver_fun w.r.t. lam evaluated at lam = 10.0.
    jac_lam = jax.jacrev(solver_fun)(10.0)
  """
  fun = lambda x, p: tree_sub(fixed_point_fun(x, p), x)
  return custom_root(fun, unpack_params, solve)


def make_kkt_optimality_fun(obj_fun, eq_fun, ineq_fun=None):
  """Makes the optimality function for KKT conditions.

  Args:
    obj_fun: objective function ``obj_fun(primal_var, params_obj)``.
    eq_fun: equality constraint function, so that
      ``eq_fun(primal_var, params_eq) == 0`` is imposed.
    ineq_fun: inequality constraint function, so that
      ``ineq_fun(primal_var, params_ineq) <= 0`` is imposed (optional).
  Returns:
    optimality_fun(x, params) where
      x = (primal_var, eq_dual_var, ineq_dual_var)
      params = (params_obj, params_eq, params_ineq)

    If ``ineq_fun`` is None, ``ineq_dual_var`` and ``params_ineq`` are
    ignored (i.e., they can be set to ``None``).
  """
  grad_fun = jax.grad(obj_fun)

  def optimality_fun(x, params):
    primal_var, eq_dual_var, ineq_dual_var = x
    params_obj, params_eq, params_ineq = params

    # Size: number of primal variables.
    _, eq_vjp_fun = jax.vjp(eq_fun, primal_var, params_eq)
    stationarity = (grad_fun(primal_var, params_obj) +
                    eq_vjp_fun(eq_dual_var)[0])

    # Size: number of equality constraints.
    primal_feasability = eq_fun(primal_var, params_eq)

    if params_ineq is not None:
      _, ineq_vjp_fun = jax.vjp(ineq_fun, primal_var, params_ineq)
      stationarity += ineq_vjp_fun(ineq_dual_var)[0]
      # Size: number of inequality constraints.
      comp_slackness = ineq_fun(primal_var, params_ineq) * ineq_dual_var
      return stationarity, primal_feasability, comp_slackness
    else:
      return stationarity, primal_feasability, None

  return optimality_fun


def make_quadratic_prog_optimality_fun():
  """Makes the optimality function for quadratic programming.

  Returns:
    optimality_fun(x, params) where
      x = (primal_var, eq_dual_var, ineq_dual_var)
      params = (params_obj, params_eq, params_ineq)
      params_obj = (Q, c)
      params_eq = (A, b)
      params_ineq = (G, h) or None
  """
  def obj_fun(primal_var, params_obj):
    Q, c = params_obj
    return (0.5 * jnp.dot(primal_var, jnp.dot(Q, primal_var)) +
            jnp.dot(primal_var, c))

  def eq_fun(primal_var, params_eq):
    A, b = params_eq
    return jnp.dot(A, primal_var) - b

  def ineq_fun(primal_var, params_ineq):
    G, h = params_ineq
    return jnp.dot(G, primal_var) - h

  return make_kkt_optimality_fun(obj_fun, eq_fun, ineq_fun)
