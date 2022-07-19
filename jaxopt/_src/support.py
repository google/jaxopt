# Copyright 2022 Google LLC
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

"""Support functions."""

from typing import Any

import jax.numpy as jnp

from jaxopt._src import tree_util


def support_all(x: Any):
  r"""Support function where all the coordinates are in the support.

  If :math:`S` is the support set, then :math:`\forall i`, :math:`x_{i} \in S`.
  
  Args:
    x: input pytree.
  Returns
    output pytree, with the same structure and dtypes as ``x``, where all the
    coordinates equal to 1.
  """
  return tree_util.tree_map(jnp.ones_like, x)


def support_nonzero(x: Any):
  r"""Support function where the support corresponds to non-zero coordinates.

  If :math:`S` is the support set, then :math:`\forall i`,

  .. math::

    x_{i} \in S \Leftrightarrow x_{i} \neq 0
  
  This support function is typically used for sparse objects with unstructured
  sparsity patterns, e.g., with the operators ``jaxopt.prox.prox_lasso`` or
  ``jaxopt.prox.prox_elastic_net``.

  Args:
    x: input pytree.
  Returns
    output pytree, with the same structure and dtypes as ``x``, equal to 1
    if ``x[i] != 0``, and 0 otherwise.
  """
  fun = lambda u: (u != 0).astype(u.dtype)
  return tree_util.tree_map(fun, x)


def support_group_nonzero(x: Any):
  r"""Support function where the support corresponds to groups of non-zero
  coordinates.

  If :math:`S` is the support set, then :math:`\forall i`,

  .. math::
    x_{i} \in S \Leftrightarrow x \not\equiv 0

  Blocks can be grouped using ``jax.vmap``. This support function is typically
  used for sparse objects with structured sparsity patterns, e.g., with the
  operator ``jaxopt.prox.prox_group_lasso``.

  Args:
    x: input pytree.
  Returns
    output pytree, with the same structure and dtypes as ``x``, where all the
    coordinates are equal to 1 if there exists an ``x[i] != 0``, and all equal
    to 0 otherwise.
  """
  fun = lambda u: jnp.any(u != 0) * jnp.ones_like(u)
  return tree_util.tree_map(fun, x)
