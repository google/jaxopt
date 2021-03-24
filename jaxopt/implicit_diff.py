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

"""Implicit differentiation of fixed point iterations."""

import functools

from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse import linalg as sparse_linalg

from jaxopt.tree_util import tree_add
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_vdot


def prox_fixed_point_vjp(
    cotangent: Any,
    fun_f: Callable,
    sol: Any,
    params_f: Any,
    prox_g: Optional[Callable] = None,
    params_g: Optional[Any] = None) -> Tuple[Any, Any]:
  """Vector-Jacobian product for the proximity operator based fixed point.

  The fixed point is:
    x = prox_g(x - grad(fun_f)(x, params_f), params_g)

  Args:
    cotangent: vector to left-multiply the Jacobian with
      (same pytree structure as `sol`).
    fun_f: a smooth function of the form fun_f(x, params_f).
    sol: solution of the fixed point.
    params_f: parameters to use for fun_f above.
    prox_g: proximity operator associated with the function g.
    params_g: parameters to use for prox_g above.
  Returns:
    (vjp_params_f, None) if prox_g is None
    (vjp_params_f, vjp_params_g) if prox_g is not None

    where vjp_params_f and vjp_params_g have the same pytree structure as
    `params_f` and `params_g`, respectively.
  """
  grad_fun = jax.grad(fun_f)
  pt = tree_sub(sol, grad_fun(sol, params_f))

  if prox_g is not None:
    prox_g = functools.partial(prox_g, scaling=1.0)

    _, vjp_prox_g = jax.vjp(prox_g, pt, params_g)

  _, vjp_grad_f = jax.vjp(grad_fun, sol, params_f)

  def f_hvp(u):
    dir_deriv = lambda x: tree_vdot(grad_fun(x, params_f), u)
    return jax.grad(dir_deriv)(sol)

  if prox_g is None:
    def matvec(u):
      # Multiply with M = M^T = B, where B = Hessian of fun_f w.r.t x.
      return f_hvp(u)
  else:
    def matvec(u):
      # Multiply with M^T u = (u^T M)^T
      # where M = AB + I - A and A = Jacobian of prox_g in first argument.
      uA = vjp_prox_g(u)[0]
      uAB = f_hvp(uA)
      return tree_sub(tree_add(uAB, u), uA)

  # The Jacobian satisfies M J = N. Computing v^T J is equivalent to
  # 1) solve M^T u = v
  # 2) compute v^T J = = u^T M J = u^T N.
  u = sparse_linalg.cg(matvec, cotangent)[0]

  if prox_g is not None:
    uA, uD = vjp_prox_g(u)

    # Compute u^T N = -u^T AC = - uA^T C,
    # where C = Jacobian of grad_f in params_f.
    vjp_params_f = tree_scalar_mul(-1, vjp_grad_f(uA)[1])

    # Compute u^T N = u^T D,
    # where D = Jacobian of prox_g in params_g.
    vjp_params_g = uD
  else:
    # Compute u^T N = -u^T AC = -u^T C.
    vjp_params_f = tree_scalar_mul(-1, vjp_grad_f(u)[1])
    vjp_params_g = None

  return vjp_params_f, vjp_params_g


def _jvp1(f, primals, tangent):
  """JVP in the first argument of f."""
  fun = lambda x: f(x, primals[1])
  return jax.jvp(fun, (primals[0],), (tangent,))[1]


def _jvp2(f, primals, tangent):
  """JVP in the second argument of f."""
  fun = lambda y: f(primals[0], y)
  return jax.jvp(fun, (primals[1],), (tangent,))[1]


def prox_fixed_point_jvp(
    tangents: Tuple[Any, Optional[Any]],
    fun_f: Callable,
    sol: Any,
    params_f: Any,
    prox_g: Optional[Callable] = None,
    params_g: Optional[Any] = None) -> Tuple[Any, Any]:
  """Vector-Jacobian product using the proximal gradient fixed point.

  The fixed point is:
    x = prox_g(x - grad(fun_f)(x, params_f), params_g)

  Args:
    tangents: a tuple containing the vectors to right-multiply the Jacobian
      with, where tangents[0] has the same pytree structure as `params_f` and
      tangents[1] is None if prox_g is None or has the same pytree structure as
      `params_g` otherwise.
    fun_f: a smooth function of the form fun_f(x, params_f).
    sol: solution of the fixed point.
    params_f: parameters to use for fun_f above.
    prox_g: proximity operator associated with the function g (optional).
    params_g: parameters to use for prox_g above (optional).
  Returns:
    (jvp_params_f, None) if prox_g is None
    (jvp_params_f, jvp_params_g) if prox_g is not None

    where `jvp_params_f` and `jvp_params_g`  are the Jacobian-vector product of
    `sol` with `tangents[0]` and `tangents[1]`, respectively. Both
    have the same pytree structure as `sol`.
  """
  grad_fun = jax.grad(fun_f)
  pt = tree_sub(sol, grad_fun(sol, params_f))

  def f_hvp(u):
    dir_deriv = lambda x: tree_vdot(grad_fun(x, params_f), u)
    return jax.grad(dir_deriv)(sol)

  if prox_g is not None:
    prox_g = functools.partial(prox_g, scaling=1.0)

    def matvec(u):
      # Multiply with M u
      # where M = AB + I - A and A = Jacobian of prox_g in first argument.
      Bu = f_hvp(u)
      ABu = _jvp1(prox_g, (pt, params_g), Bu)
      Au = _jvp1(prox_g, (pt, params_g), u)
      return tree_sub(tree_add(ABu, u), Au)

    # Compute Nv = -AC v = - A Cv,
    # where C = Jacobian of grad_f in params_f.
    Cv = _jvp2(grad_fun, (sol, params_f), tangents[0])
    ACv = _jvp1(prox_g, (pt, params_g), Cv)
    minus_ACv = tree_scalar_mul(-1, ACv)
    jvp_params_f = sparse_linalg.cg(matvec, minus_ACv)[0]

    # Compute Nv = Dv,
    # where D = Jacobian of prox_g in params_g.
    Dv = _jvp2(prox_g, (pt, params_g), tangents[1])
    jvp_params_g = sparse_linalg.cg(matvec, Dv)[0]

  else:
    def matvec(u):
      # Multiply with M = B, where B = Hessian of fun_f w.r.t x.
      return f_hvp(u)

    # Compute Nv = -AC v = -Cv.
    Cv = _jvp2(grad_fun, (sol, params_f), tangents[0])
    minus_Cv = tree_scalar_mul(-1, Cv)
    jvp_params_f = sparse_linalg.cg(matvec, minus_Cv)[0]
    jvp_params_g = None

  return jvp_params_f, jvp_params_g
