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

"""Projection operators."""

from typing import Any

import jax
import jax.numpy as jnp
from jaxopt import tree_util

from jaxopt import root_finding


@jax.custom_jvp
def _projection_unit_simplex(x: jnp.ndarray) -> jnp.ndarray:
  """Projection onto the unit simplex."""
  s = 1.0
  n_features = x.shape[0]
  u = jnp.sort(x)[::-1]
  cssv = jnp.cumsum(u) - s
  ind = jnp.arange(n_features) + 1
  cond = u - cssv / ind > 0
  idx = jnp.count_nonzero(cond)
  threshold = cssv[idx - 1] / idx.astype(x.dtype)
  return jax.nn.relu(x - threshold)


@_projection_unit_simplex.defjvp
def _projection_unit_simplex_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  primal_out = _projection_unit_simplex(x)
  supp = primal_out > 0
  card = jnp.count_nonzero(supp)
  tangent_out = supp * x_dot - (jnp.dot(supp, x_dot) / card) * supp
  return primal_out, tangent_out


def projection_simplex(x: jnp.ndarray, s: float = 1.0) -> jnp.ndarray:
  r"""Projection onto the simplex.

  The output is ``argmin_{p : 0 <= p <= s, jnp.sum(p) = s} ||x - p||``

  Args:
    x: vector to project, an array of shape (n,).
    s: value p should sum to (default: 1.0).
  Returns:
    p: projected vector, an array of shape (n,).
  """
  return s * _projection_unit_simplex(x / s)


def projection_l2_sphere(x: Any, diam: float = 1.0) -> Any:
  r"""Projection onto the sphere, { x | || x || = diam}

  The output is: ``argmin_{y, ||y||= diam} 0.5 ||y - x||^2 = diam * x / ||x||``.

  Args:
    x: jnp.ndarray to project.
    diam: diameter of the sphere.

  Returns:
    y: output jnp.ndarray (same size as x) normalized to have suitable norm.
  """
  factor = diam / tree_util.tree_l2_norm(x)
  return tree_util.tree_scalar_mul(factor, x)


def projection_box_section(x, params, check_feasible=False):
  """Projection onto a box section.

  The projection is::

    argmin_{p : alpha_i <= p_i <= beta_i, jnp.dot(w, p) = c} ||x - p||

  where ``(alpha, beta, w, c) = params``.

  Args:
    x: vector to project, an array of shape (n,).
    params: tuple of parameters, ``params = (alpha, beta, w, c)``, where
      ``w`` is a positive vector. The problem is infeasible if
      dot(w, alpha) > c or if dot(w, beta) < c.
    check_feasible: whether to check feasibility or not.
      If True, function cannot be jitted.
  Returns:
    p: projected vector, an array of shape (n,).
  """
  # An optimal solution has the form
  # p_i = clip(w_i * tau + x_i, alpha_i, beta_i) for all i
  # where tau is the root of fun(tau, params) = dot(w, p) - c = 0.
  def root(x, params):
    def fun(tau, args):
      x, params = args
      alpha, beta, w, c = params
      p = jnp.clip(w * tau + x, alpha, beta)
      return jnp.dot(w, p) - c

    alpha, beta, w, _ = params
    lower = jax.lax.stop_gradient(jnp.min((alpha - x) / w))
    upper = jax.lax.stop_gradient(jnp.max((beta - x) / w))
    bisect_fun = root_finding.make_bisect_fun(fun=fun, lower=lower, upper=upper,
                                              increasing=True)
    args = (x, params)
    return bisect_fun(args)

  alpha, beta, w, c = params

  if check_feasible:
    if jnp.dot(w, alpha) > c:
      raise ValueError("alpha should satisfy dot(w, alpha) <= c")

    if jnp.dot(w, beta) < c:
      raise ValueError("beta should satisfy dot(w, beta) >= c")

  return jnp.clip(w * root(x, params) + x, alpha, beta)
