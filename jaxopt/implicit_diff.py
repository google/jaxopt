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

from typing import Callable
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse import linalg as sparse_linalg


def prox_fixed_point_vjp(
    cotangent: jnp.ndarray,
    fun_f: Callable,
    sol: jnp.ndarray,
    params_f: jnp.ndarray,
    prox_g: Optional[Callable] = None,
    params_g: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
  pt = sol - grad_fun(sol, params_f)
  if params_g is None:
    params_g = 1.0

  if prox_g is not None:
    _, vjp_prox_g = jax.vjp(prox_g, pt, params_g)

  _, vjp_grad_f = jax.vjp(grad_fun, sol, params_f)

  def f_hvp(u):
    dir_deriv = lambda x: jnp.vdot(grad_fun(x, params_f), u)
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
      return jnp.transpose(uAB + u - uA)

  # The Jacobian satisfies M J = N. Computing v^T J is equivalent to
  # 1) solve M^T u = v
  # 2) compute v^T J = = u^T M J = u^T N.
  u = sparse_linalg.cg(matvec, cotangent)[0]

  if prox_g is not None:
    uA, uD = vjp_prox_g(u)

    # Compute u^T N = -u^T AC = - uA^T C,
    # where C = Jacobian of grad_f in params_f.
    vjp_params_f = -vjp_grad_f(uA)[1]

    # Compute u^T N = u^T D,
    # where D = Jacobian of prox_g in params_g.
    vjp_params_g = uD
  else:
    # Compute u^T N = -u^T AC = -u^T C.
    vjp_params_f = -vjp_grad_f(u)[1]
    vjp_params_g = None

  return vjp_params_f, vjp_params_g
