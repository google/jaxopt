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


def fixed_point_vjp(fixed_point_fun: Callable,
                    sol: Any,
                    params: Any,
                    cotangent: Any) -> Any:
  """Vector-Jacobian product of a fixed point.

  The fixed point is `sol = fixed_point_fun(sol, params)`.

  Args:
    fixed_point_fun: the fixed point function to use.
    sol: solution of the fixed point (pytree).
    params: parameters to use for `fixed_point_fun` above (pytree).
    cotangent: vector to left-multiply the Jacobian with
      (pytree, same structure as `sol`).
  Returns:
    Vector-Jacobian product w.r.t. `params` of `sol` with cotangent.
    It has the same pytree  structure as `params`.
  """
  _, vjp_fun = jax.vjp(fixed_point_fun, sol, params)

  def matvec(u):
    # Compute the multiplication A^T u = (u^T A)^T = (u^T (Id - J))^T.
    uJ = vjp_fun(u)[0]
    return tree_sub(u, uJ)

  # The solution of A^T u = v, where A = Id - J, v = cotangent, and
  # J = jacobian(fixed_point_fun, argnums=0).
  u = sparse_linalg.cg(matvec, cotangent)[0]

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
                    sol: Any,
                    params: Any,
                    tangent: Any) -> Any:
  """Jacobian-vector product of a fixed point.

  The fixed point is `sol = fixed_point_fun(sol, params)`.

  Args:
    fixed_point_fun: the fixed point function to use.
    sol: solution of the fixed point (pytree).
    params: parameters to use for `fixed_point_fun` above (pytree).
    tangent: a pytree to right-multiply the Jacobian with, with the same pytree
      structure as `params`.
  Returns:
    Jacobian-vector product w.r.t. `params` of `sol` with `tangent`.
    It has the same pytree structure as `sol`.
  """
  def matvec(u):
    # Compute the multiplication Au = (Id - J)u,
    # where A = jacobian(fixed_point_fun, argnums=0).
    Ju = _jvp1(fixed_point_fun, (sol, params), u)
    return tree_sub(u, Ju)

  Jv = _jvp2(fixed_point_fun, (sol, params), tangent)
  return sparse_linalg.cg(matvec, Jv)[0]


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


def _custom_fixed_point(solver_fun, fixed_point_fun):
  def solver_fun_fwd(params):
    sol = solver_fun(params)
    return sol, (params, sol)

  def solver_fun_bwd(res, cotangent):
    params, sol = res
    vjp = fixed_point_vjp(fixed_point_fun=fixed_point_fun,
                          sol=sol, params=params, cotangent=cotangent)
    return (vjp,)

  wrapped_solver_fun = jax.custom_vjp(solver_fun)
  wrapped_solver_fun.defvjp(solver_fun_fwd, solver_fun_bwd)

  return wrapped_solver_fun


def custom_fixed_point(fixed_point_fun):
  """Decorator for adding implicit differentiation to a solver.

  Args:
    fixed_point_fun: a fixed point function, `fixed_point_fun(x, params)`.
      The invariant is `sol == fixed_point_fun(sol, params)` at the fixed point
      `sol`.

  Returns:
    solver function wrapped with implicit differentiation.

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
  def wrapper(solver_fun):
    return _custom_fixed_point(solver_fun, fixed_point_fun)
  return wrapper
