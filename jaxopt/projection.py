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
