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
"""Interface for linear operators."""

import functools
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from jaxopt.tree_util import tree_map


class DenseLinearOperator:
  def __init__(self, pytree):
    self.pytree = pytree

  def __call__(self, x):
    return self.matvec(x)

  def matvec(self, x):
    return tree_map(jnp.dot, self.pytree, x)

  def rmatvec(self, _, y):
    return tree_map(lambda w, yi: jnp.dot(w.T, yi), self.pytree, y)

  def matvec_and_rmatvec(self, x, y):
    return self.matvec(x), self.rmatvec(x, y)

  def normal_matvec(self, x):
    """Computes A^T A x."""
    return self.rmatvec(x, self.matvec(x))

  def diag(self):
    diags_only = tree_map(jnp.diag, self.pytree)
    return diags_only

  def columns_l2_norms(self, squared=False):
    def col_norm(w):
      col_norms = jnp.sum(jnp.square(w), axis=0)
      if not squared:
        col_norms = jnp.sqrt(col_norms)
      return col_norms

    return tree_map(col_norm, self.pytree)


class FunctionalLinearOperator:
  def __init__(self, fun, params):
    self.fun = functools.partial(fun, params)

  def __call__(self, x):
    return self.matvec(x)

  def matvec(self, x):
    return self.fun(x)

  def rmatvec(self, x, y):
    return self.matvec_and_rmatvec(x, y)[1]

  def matvec_and_rmatvec(self, x, y):
    matvec_x, vjp = jax.vjp(self.matvec, x)
    (rmatvec_y,) = vjp(y)
    return matvec_x, rmatvec_y

  def normal_matvec(self, x):
    """Computes A^T A x from matvec(x) = A x."""
    matvec_x, vjp = jax.vjp(self.matvec, x)
    return vjp(matvec_x)[0]


def _make_linear_operator(matvec):
  if matvec is None:
    return DenseLinearOperator
  else:
    return functools.partial(FunctionalLinearOperator, matvec)


def block_row_matvec(block, x):
  """Performs a matvec for a row of block matrices.
  
  The following matvec is performed: 
  [U1, ..., UN] * [x1, ..., xN]
  where U1, ..., UN are matrices and x1, ..., xN are vectors
  of compatible shapes.
  """
  if len(block) != len(x):
    raise ValueError(
      "We need as many blocks in the matrix as in the vector."
      )
  return sum(jax.tree_util.tree_map(jnp.dot, block, x))


# TODO(gnegiar): Extend to arbitrary block shapes.
@jax.tree_util.register_pytree_node_class
@dataclass
class BlockLinearOperator:
  """Represents a linear operator defined by blocks over a block pytree.

  Attributes:
    blocks: a 2x2 block matrix of the form
      [[A, B]
       [C, D]]
  """

  blocks: Tuple[Tuple[jnp.array]]

  def __call__(self, x):
    return self.matvec(x)

  def matvec(self, x):
    """Performs the block matvec with u defined by blocks.

    The matvec is of form:
               [u1, u2]
    [[A, B]  *
     [C, D]]

    """
    return jax.tree_util.tree_map(
      lambda row_of_blocks: block_row_matvec(row_of_blocks, x),
      self.blocks,
      is_leaf=lambda x: x is self.blocks[0] or x is self.blocks[1],
    )

  def rmatvec(self, x, y):
    return self.matvec_and_rmatvec(x, y)[1]

  def matvec_and_rmatvec(self, x, y):
    matvec_x, vjp = jax.vjp(self.matvec, x)
    (rmatvec_y,) = vjp(y)
    return matvec_x, rmatvec_y

  def normal_matvec(self, x):
    """Computes A^T A x from matvec(x) = A x."""
    matvec_x, vjp = jax.vjp(self.matvec, x)
    return vjp(matvec_x)[0]

  def tree_flatten(self):
    return self.blocks, None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(children)
