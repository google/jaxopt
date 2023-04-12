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

"""Proximity operators."""

from typing import Any
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp

from jaxopt._src import tree_util


def prox_none(x: Any,
              hyperparams: Optional[Any] = None,
              scaling: float = 1.0) -> Any:
  r"""Proximal operator for :math:`g(x) = 0`, i.e., the identity function.

  Since :math:`g(x) = 0`, the output is:

  .. math::

    \underset{y}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2 = x

  Args:
    x: input pytree.
    hyperparams: ignored.
    scaling: ignored.
  Returns:
    output pytree, with the same structure as ``x``.
  """
  del hyperparams, scaling
  return x


def prox_lasso(x: Any,
               l1reg: Optional[Any] = None,
               scaling: float = 1.0) -> Any:
  r"""Proximal operator for the l1 norm, i.e., soft-thresholding operator.

  .. math::

    \underset{y}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
    + \text{scaling} \cdot \text{l1reg} \cdot ||y||_1

  When ``l1reg`` is a pytree, the weights are applied coordinate-wise.

  Args:
    x: input pytree.
    l1reg: regularization strength, float or pytree with the same structure
      as ``x``.
    scaling: a scaling factor.

  Returns:
    output pytree, with same structure as ``x``.
  """
  if l1reg is None:
    l1reg = 1.0

  if type(l1reg) == float:
    l1reg = tree_util.tree_map(lambda y: l1reg*jnp.ones_like(y), x)

  def fun(u, v): return jnp.sign(u) * jax.nn.relu(jnp.abs(u) - v * scaling)
  return tree_util.tree_map(fun, x, l1reg)

def prox_non_negative_lasso(x: Any,
                            l1reg: Optional[float] = None,
                            scaling: float = 1.0) -> Any:
  r"""Proximal operator for the l1 norm on the non-negative orthant.

  .. math::

    \underset{y \ge 0}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
    + \text{scaling} \cdot \text{l1reg} \cdot ||y||_1

  Args:
    x: input pytree.
    l1reg: regularization strength.
    scaling: a scaling factor.

  Returns:
    output pytree, with the same structure as ``x``.
  """
  if l1reg is None:
    l1reg = 1.0

  pytree = tree_util.tree_map(lambda y: y - l1reg*scaling, x)
  return tree_util.tree_map(jax.nn.relu, pytree)


def prox_elastic_net(x: Any,
                     hyperparams: Optional[Tuple[Any, Any]] = None,
                     scaling: float = 1.0) -> Any:
  r"""Proximal operator for the elastic net.

  .. math::

    \underset{y}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
    + \text{scaling} \cdot \text{hyperparams[0]} \cdot g(y)

  where :math:`g(y) = ||y||_1 + \text{hyperparams[1]} \cdot 0.5 \cdot ||y||_2^2`.

  Args:
    x: input pytree.
    hyperparams: a tuple, where both ``hyperparams[0]`` and ``hyperparams[1]``
      can be either floats or pytrees with the same structure as ``x``.
    scaling: a scaling factor.

  Returns:
    output pytree, with the same structure as ``x``.
  """
  if hyperparams is None:
    hyperparams = (1.0, 1.0)

  lam = tree_util.tree_map(lambda y: hyperparams[0]*jnp.ones_like(
      y), x) if type(hyperparams[0]) == float else hyperparams[0]
  gam = tree_util.tree_map(lambda y: hyperparams[1]*jnp.ones_like(
      y), x) if type(hyperparams[1]) == float else hyperparams[1]

  def prox_l1(u, lambd): return jnp.sign(u) * jax.nn.relu(jnp.abs(u) - lambd)

  def fun(u, lambd, gamma): return (prox_l1(u, scaling * lambd) /
                                    (1.0 + scaling * lambd * gamma))
  return tree_util.tree_map(fun, x, lam, gam)


def prox_group_lasso(x: Any,
                     l2reg: Optional[float] = 1.0,
                     scaling=1.0) -> Any:
  r"""Proximal operator for the l2 norm, i.e., block soft-thresholding operator.

  .. math::

    \underset{y}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
    + \text{scaling} \cdot \text{l2reg} \cdot ||y||_2

  Blocks can be grouped using ``jax.vmap``.

  Args:
    x: input pytree.
    l2reg: regularization strength.
    scaling: a scaling factor.

  Returns:
    output pytree, with the same structure as ``x``.
  """
  if l2reg is None:
    l2reg = 1.0

  l2_norm = tree_util.tree_l2_norm(x)
  factor = 1 - l2reg * scaling / l2_norm
  factor = jnp.where(factor >= 0, factor, 0)
  return tree_util.tree_scalar_mul(factor, x)


def prox_ridge(x: Any,
               l2reg: Optional[float] = 1.0,
               scaling=1.0) -> Any:
  r"""Proximal operator for the squared l2 norm.

  .. math::

    \underset{y}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
    + \text{scaling} \cdot \text{l2reg} \cdot ||y||_2^2

  Args:
    x: input pytree.
    l2reg: regularization strength.
    scaling: a scaling factor.

  Returns:
    output pytree, with the same structure as ``x``.
  """
  if l2reg is None:
    l2reg = 1.0

  factor = 1. / (1 + scaling * l2reg)
  return tree_util.tree_scalar_mul(factor, x)


def prox_non_negative_ridge(x: Any,
                            l2reg: Optional[float] = 1.0,
                            scaling: float = 1.0):
  r"""Proximal operator for the squared l2 norm on the non-negative orthant.

  .. math::

    \underset{y \ge 0}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
    + \text{scaling} \cdot \text{l2reg} \cdot ||y||_2^2

  Args:
    x: input pytree.
    l2reg: regularization strength.
    scaling: a scaling factor.

  Returns:
    output pytree, with the same structure as ``x``.
  """
  if l2reg is None:
    l2reg = 1.0

  pytree = tree_util.tree_scalar_mul(1./ (1 + l2reg * scaling), x)
  return tree_util.tree_map(jax.nn.relu, pytree)


def make_prox_from_projection(projection):
  """Transforms a projection into a proximal operator."""
  def prox(x, hyperparams=None, scaling=1.0):
    del scaling  # The scaling parameter is meaningless for projections.
    return projection(x, hyperparams)
  return prox
