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
import itertools
import operator

import jax
from jax import tree_util as tu
import jax.numpy as jnp
import numpy as onp


tree_flatten = tu.tree_flatten
tree_leaves = tu.tree_leaves
tree_map = tu.tree_map
tree_reduce = tu.tree_reduce
tree_unflatten = tu.tree_unflatten


def broadcast_pytrees(*trees):
  """Broadcasts leaf pytrees to match treedef shared by the other arguments.

  Args:
    *trees: A `Sequence` of pytrees such that all elements that are *not* leaf
      pytrees (i.e. single arrays) have the same treedef.

  Returns:
    The input `Sequence` of pytrees `*trees` with leaf pytrees (i.e. single
    arrays) replaced by pytrees matching the treedef of non-shallow elements via
    broadcasting.

  Raises:
    ValueError: If two or more pytrees in `*trees` that are not leaf pytrees
      differ in their structure (treedef).
  """
  leaves, treedef, is_leaf = [], None, []
  for tree in trees:
    leaves_i, treedef_i = tu.tree_flatten(tree)
    is_leaf_i = tu.treedef_is_leaf(treedef_i)
    if not is_leaf_i:
      treedef = treedef or treedef_i
      if treedef_i != treedef:
        raise ValueError('Pytrees are not broadcastable.: '
                         f'{treedef} != {treedef_i}')
    leaves.append(leaves_i)
    is_leaf.append(is_leaf_i)
  if treedef is not None:
    max_num_leaves = max(len(leaves_i) for leaves_i in leaves)
    broadcast_leaf = lambda leaf: itertools.repeat(leaf[0], max_num_leaves)
    leaves = [broadcast_leaf(leaves_i) if is_leaf_i else leaves_i
              for (leaves_i, is_leaf_i) in zip(leaves, is_leaf)]
    return tuple(treedef.unflatten(leaves_i) for leaves_i in leaves)
  # All Pytrees are leaves.
  return trees


tree_add = functools.partial(tree_map, operator.add)
tree_add.__doc__ = "Tree addition."

tree_sub = functools.partial(tree_map, operator.sub)
tree_sub.__doc__ = "Tree subtraction."

tree_mul = functools.partial(tree_map, operator.mul)
tree_mul.__doc__ = "Tree multiplication."

tree_div = functools.partial(tree_map, operator.truediv)
tree_div.__doc__ = "Tree division."


def tree_scalar_mul(scalar, tree_x):
  """Compute scalar * tree_x."""
  return tree_map(lambda x: scalar * x, tree_x)


def tree_add_scalar_mul(tree_x, scalar, tree_y):
  """Compute tree_x + scalar * tree_y."""
  return tree_map(lambda x, y: x + scalar * y, tree_x, tree_y)


_vdot = functools.partial(jnp.vdot, precision=jax.lax.Precision.HIGHEST)


def _vdot_safe(a, b):
  return _vdot(jnp.asarray(a), jnp.asarray(b))


def tree_vdot(tree_x, tree_y):
  """Compute the inner product <tree_x, tree_y>."""
  vdots = tree_map(_vdot_safe, tree_x, tree_y)
  return tree_reduce(operator.add, vdots)


def _vdot_real(x, y):
  """Vector dot-product guaranteed to have a real valued result despite
     possibly complex input. Thus neglects the real-imaginary cross-terms.
     The result is a real float.
  """
  #result = _vdot(x.real, y.real)
  #if jnp.iscomplexobj(x) and jnp.iscomplexobj(y):
  #  result += _vdot(x.imag, y.imag)
  result = _vdot(x, y).real  # NOTE: without jit this is faster than variant above, no difference with jit
  return result


def tree_vdot_real(tree_x, tree_y):
  """Compute the real part of the inner product <tree_x, tree_y>."""
  return sum(tree_leaves(tree_map(_vdot_real, tree_x, tree_y)))


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
  squared_tree = tree_map(lambda leaf: jnp.square(leaf.real) + jnp.square(leaf.imag), tree_x)
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
  leaves_vec = tree_leaves(tree_map(jnp.ravel, tree_x))
  return jnp.max(jnp.abs(jnp.concatenate(leaves_vec)))


def tree_where(cond, a, b):
  """jnp.where for trees.

  Mimic broadcasting semantic of jnp.where.
  cond, a and b can be arrays (including scalars) broadcastable to the leaves of
  the other input arguments.

  Args:
    cond: pytree of booleans arrays, or single array broadcastable to the shapes
      of leaves of `a` and `b`.
    a: pytree of arrays, or single array broadcastable to the shapes of leaves
      of `cond` and `b`.
    b: pytree of arrays, or single array broadcastable to the shapes of leaves
      of `cond` and `a`.

  Returns:
    pytree of arrays, or single array
  """
  cond, a, b = broadcast_pytrees(cond, a, b)
  return tree_map(jnp.where, cond, a, b)


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


def tree_single_dtype(tree, convert_in_jax_dtype=True):
  """The dtype for all values in a tree, provided that all leaves share the same type.

      If the leaves have different type, raise a ValueError.

      Args:
        tree: tree to get the dtype of
        convert_in_jax_type: whether to convert the types in JAX precision.
          Namely, a numpy int64 type is converted in a jax.numpy int32 type
          by default unless one enables double precision using
          jax.config.update("jax_enable_x64", True)
      Return:
        dtype shared by all leaves of the tree
      """
  if convert_in_jax_dtype:
    dtypes = set(
      jnp.asarray(p).dtype
      for p in tu.tree_leaves(tree)
      if isinstance(
        p, (bool, int, float, complex, onp.ndarray, jnp.ndarray)
      )
    )
  else:
    dtypes = set(
      onp.asarray(p).dtype
      for p in tu.tree_leaves(tree)
      if isinstance(
        p, (bool, int, float, complex, onp.ndarray, jnp.ndarray)
      )
    )
  if not dtypes:
    return None
  if len(dtypes) == 1:
    dtype = dtypes.pop()
    return dtype
  raise ValueError("Found more than one dtype in the tree.")

def get_real_dtype(dtype):
  """Dtype corresponding of real part of a complex dtype."""
  if dtype not in [f'complex{i}' for i in [4, 8, 16, 32, 64, 128]]:
    return dtype
  else:
    return dtype.type(0).real.dtype


def tree_conj(tree):
  """Complex conjugate of a tree."""
  return tree_map(jnp.conj, tree)


def tree_real(tree):
  """Real part of a tree"""
  return tree_map(jnp.real, tree)


def tree_imag(tree):
  """Imaginary part of a tree"""
  return tree_map(jnp.imag, tree)

