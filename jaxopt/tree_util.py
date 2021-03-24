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

"""Tree utilities."""

import functools
import operator

import jax
from jax import tree_util as tu
import jax.numpy as jnp

tree_map = tu.tree_map
tree_multimap = tu.tree_multimap
tree_reduce = tu.tree_reduce

tree_add = functools.partial(tree_multimap, operator.add)
tree_sub = functools.partial(tree_multimap, operator.sub)
tree_mul = functools.partial(tree_multimap, operator.mul)


def tree_scalar_mul(scalar, tree_x):
  """Compute scalar * tree_x."""
  return tree_map(lambda x: scalar * x, tree_x)


def tree_add_scalar_mul(tree_x, scalar, tree_y):
  """Compute tree_x + scalar * tree_y."""
  return tree_multimap(lambda x, y: x + scalar * y, tree_x, tree_y)


_vdot = functools.partial(jnp.vdot, precision=jax.lax.Precision.HIGHEST)


def tree_vdot(tree_x, tree_y):
  """Compute the inner product <tree_x, tree_y>."""
  vdots = tree_multimap(_vdot, tree_x, tree_y)
  return tree_reduce(operator.add, vdots)


def tree_sum(tree_x):
  """Compute sum(tree_x)."""
  sums = tree_map(jnp.sum, tree_x)
  return tree_reduce(operator.add, sums)


def tree_l2_norm(tree_x, squared=False):
  """Compute the l2 norm ||tree_x||."""
  squared_tree = tree_map(jnp.square, tree_x)
  sqnorm = tree_sum(squared_tree)
  if squared:
    return sqnorm
  else:
    return jnp.sqrt(sqnorm)
