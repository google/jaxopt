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
import numpy as onp


tree_flatten = tu.tree_flatten
tree_leaves = tu.tree_leaves
tree_map = tu.tree_map
tree_multimap = tu.tree_multimap
tree_reduce = tu.tree_reduce
tree_unflatten = tu.tree_unflatten

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


def _vdot_safe(a, b):
  return _vdot(jnp.asarray(a), jnp.asarray(b))

def tree_vdot(tree_x, tree_y):
  """Compute the inner product <tree_x, tree_y>."""
  vdots = tree_multimap(_vdot_safe, tree_x, tree_y)
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


def tree_zeros_like(tree_x):
  """Creates an all-zero tree with the same structure as tree_x."""
  return tree_map(jnp.zeros_like, tree_x)


def tree_collapse(tree):
  """Takes a pytree of arrays in input and return 1D vector of flattened leaves.

  Reciprocal function of ``tree_uncollapse``. 
  
  Args:
    tree: tree of arrays
  Return type:
    array
  Returns:
    a 1D vector
  """
  leaves = tree_leaves(tree)
  leaves = list(map(lambda params: params.flatten(), leaves))
  return jnp.concatenate(leaves, axis=0)


def tree_uncollapse(tree, vec):
  """Returns a pytree with same elements as 1D vector ``vec`` and with same structure as ``tree``.

  Reciprocal function of ``tree_collapse``. Transform the 1D vector into a pytree
  compatible with ``tree`` structure, with same shapes.
  
  Args:
    tree: tree of arrays
    vec: 1D vector
  Return type:
    pytree of arrays
  Returns:
    a pytree with same structure as ``tree``
  """
  leaves, tree_def = tree_flatten(tree)
  sizes =  list(map(lambda params: params.size, leaves))
  shapes = list(map(lambda params: params.shape, leaves))
  indices = onp.cumsum(onp.array(sizes, dtype=onp.int32))
  vecs = jnp.split(vec, indices)
  vecs = list(map(lambda v,shape: v.reshape(shape), vecs, shapes))
  return tree_unflatten(tree_def, vecs)

