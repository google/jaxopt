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
tree_add.__doc__ = "Tree addition."

tree_sub = functools.partial(tree_multimap, operator.sub)
tree_sub.__doc__ = "Tree subtraction."

tree_mul = functools.partial(tree_multimap, operator.mul)
tree_mul.__doc__ = "Tree multiplication."

tree_div = functools.partial(tree_multimap, operator.truediv)
tree_div.__doc__ = "Tree division."


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


def tree_dot(tree_x, tree_y):
  """Compute leaves-wise dot product between pytree of arrays.

  Useful to store block diagonal linear operators: each leaf of the tree
  corresponds to a block."""
  return tree_map(jnp.dot, tree_x, tree_y)


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


def tree_ones_like(tree_x):
  """Creates an all-ones tree with the same structure as tree_x."""
  return tree_map(jnp.ones_like, tree_x)


def tree_average(trees, weights):
  """Return the linear combination of a list of trees.

  Args:
    trees: tree of arrays with shape (m,...)
    weights: array of shape (m,)

  Returns:
    a single tree that is the linear combination of all trees
  """
  return tree_map(lambda x: jnp.tensordot(weights, x, axes=1), trees)


def tree_gram(a):
  """Compute Gramn matrix from the pytree of batchs of vectors.

  Args:
    a: pytree of arrays of shape (m,...)

  Returns:
    arrays of shape (m,m) of all dot products
  """
  vmap_left = jax.vmap(tree_vdot, in_axes=(0,None))
  vmap_right = jax.vmap(vmap_left, in_axes=(None,0))
  return vmap_right(a, a)


def tree_inf_norm(tree_x):
  """Computes the infinity norm of a pytree."""
  leaf_inf_norm = tree_map(lambda x: jnp.max(jnp.abs(x)), tree_x)
  return tree_reduce(jnp.maximum, leaf_inf_norm)


def tree_where(cond, a, b):
  """jnp.where for trees.
  
  Mimic broadcasting semantic of jnp.where.
  a and b can be arrays (including scalars) broadcastable to the leaves of cond.
  
  Args:
    cond: pytree of booleans arrays.
    a   : pytree of arrays, or single array broadcastable
      to the shapes of leaves of cond.
    b   : pytree of arrays, or single array broadcastable
      to the shapes of leaves of cond.
    
  Returns:
    pytree of arrays, or single array
  """
  a_scalar = tu.treedef_is_leaf(tu.tree_structure(a))
  b_scalar = tu.treedef_is_leaf(tu.tree_structure(b))
  if a_scalar and b_scalar:
    return tree_map(lambda c: jnp.where(c, a, b), cond)
  if b_scalar:
    return tree_map(lambda c, u: jnp.where(c, u, b), cond, a)
  if a_scalar:
    return tree_map(lambda c, v: jnp.where(c, a, v), cond, b)
  return tree_map(lambda c, u, v: jnp.where(c, u, v), cond, a, b)


def tree_negative(tree):
  """Computes elementwise negation -x."""
  return tree_scalar_mul(-1, tree)


def tree_reciproqual(tree):
  """Computes elementwise inverse 1/x."""
  return tree_map(lambda x: jnp.reciprocal(x), tree)


def tree_mean(tree):
  """Mean reduction for trees."""
  leaves_avg = tree_map(jnp.mean, tree)
  return tree_sum(leaves_avg) / len(tree_leaves(leaves_avg))
