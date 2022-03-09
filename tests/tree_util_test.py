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

from absl.testing import absltest

import jax.numpy as jnp
import jax.tree_util as jtu

from jaxopt import tree_util
from jaxopt._src import test_util

import numpy as onp


class TreeUtilTest(test_util.JaxoptTestCase):

  def setUp(self):
    super().setUp()
    rng = onp.random.RandomState(0)

    self.tree_A = (rng.randn(20, 10), rng.randn(20))
    self.tree_B = (rng.randn(20, 10), rng.randn(20))

    self.tree_A_dict = (1.0, {"k1": 1.0, "k2": (1.0, 1.0)}, 1.0)
    self.tree_B_dict = (1.0, {"k1": 2.0, "k2": (3.0, 4.0)}, 5.0)

    self.array_A = rng.randn(20)
    self.array_B = rng.randn(20)

  def _assertTreesAllClose(self, tree_A, tree_B):
    self.assertArraysAllClose(tree_A[0], tree_B[0])
    self.assertArraysAllClose(tree_A[1], tree_B[1])

  def test_tree_add(self):
    expected = self.array_A + self.array_B
    got = tree_util.tree_add(self.array_A, self.array_B)
    self._assertTreesAllClose(expected, got)

    expected = (self.tree_A[0] + self.tree_B[0],
                self.tree_A[1] + self.tree_B[1])
    got = tree_util.tree_add(self.tree_A, self.tree_B)
    self._assertTreesAllClose(expected, got)

  def test_tree_sub(self):
    expected = self.array_A - self.array_B
    got = tree_util.tree_sub(self.array_A, self.array_B)
    self._assertTreesAllClose(expected, got)

    expected = (self.tree_A[0] - self.tree_B[0],
                self.tree_A[1] - self.tree_B[1])
    got = tree_util.tree_sub(self.tree_A, self.tree_B)
    self._assertTreesAllClose(expected, got)

  def test_tree_mul(self):
    expected = self.array_A * self.array_B
    got = tree_util.tree_mul(self.array_A, self.array_B)
    self._assertTreesAllClose(expected, got)

    expected = (self.tree_A[0] * self.tree_B[0],
                self.tree_A[1] * self.tree_B[1])
    got = tree_util.tree_mul(self.tree_A, self.tree_B)
    self._assertTreesAllClose(expected, got)

  def test_tree_scalar_mul(self):
    expected = 0.5 * self.array_A
    got = tree_util.tree_scalar_mul(0.5, self.array_A)
    self._assertTreesAllClose(expected, got)

    expected = (0.5 * self.tree_A[0], 0.5 * self.tree_A[1])
    got = tree_util.tree_scalar_mul(0.5, self.tree_A)
    self._assertTreesAllClose(expected, got)

  def test_tree_add_scalar_mul(self):
    expected = (self.tree_A[0] + 0.5 * self.tree_B[0],
                self.tree_A[1] + 0.5 * self.tree_B[1])
    got = tree_util.tree_add_scalar_mul(self.tree_A, 0.5, self.tree_B)
    self._assertTreesAllClose(expected, got)

  def test_tree_vdot(self):
    expected = jnp.vdot(self.array_A, self.array_B)
    got = tree_util.tree_vdot(self.array_A, self.array_B)
    self.assertAllClose(expected, got)

    expected = 15.0
    got = tree_util.tree_vdot(self.tree_A_dict, self.tree_B_dict)
    self.assertAllClose(expected, got)

    expected = (jnp.vdot(self.tree_A[0], self.tree_B[0]) +
                jnp.vdot(self.tree_A[1], self.tree_B[1]))
    got = tree_util.tree_vdot(self.tree_A, self.tree_B)
    self.assertAllClose(expected, got)

  def test_tree_div(self):
    expected = (self.tree_A[0] / self.tree_B[0], self.tree_A[1] / self.tree_B[1])
    got = tree_util.tree_div(self.tree_A, self.tree_B)
    self.assertAllClose(expected, got)

    got = tree_util.tree_div(self.tree_A_dict, self.tree_B_dict)
    expected = (1.0, {'k1': 0.5, 'k2': (0.333333333, 0.25)}, 0.2)
    self.assertAllClose(expected, got)

  def test_tree_sum(self):
    expected = jnp.sum(self.array_A)
    got = tree_util.tree_sum(self.array_A)
    self.assertAllClose(expected, got)

    expected = (jnp.sum(self.tree_A[0]) + jnp.sum(self.tree_A[1]))
    got = tree_util.tree_sum(self.tree_A)
    self.assertAllClose(expected, got)

  def test_tree_l2_norm(self):
    expected = jnp.sqrt(jnp.sum(self.array_A ** 2))
    got = tree_util.tree_l2_norm(self.array_A)
    self.assertAllClose(expected, got)

    expected = jnp.sqrt(jnp.sum(self.tree_A[0] ** 2) +
                        jnp.sum(self.tree_A[1] ** 2))
    got = tree_util.tree_l2_norm(self.tree_A)
    self.assertAllClose(expected, got)

  def test_tree_zeros_like(self):
    tree = tree_util.tree_zeros_like(self.tree_A)
    self.assertAllClose(tree_util.tree_l2_norm(tree), 0.0)

  def test_tree_where(self):
    c = jnp.array([True, False, True, False]), jnp.array([True, False, False, True])
    a = jnp.array([1., 2., 3., 4.]), jnp.array([5., 6., 7., 8.])
    b = jnp.array(42.), jnp.array(13.)
    d = tree_util.tree_where(c, a, b)
    self.assertAllClose(d[0], jnp.array([1., 42., 3., 42.]))
    self.assertAllClose(d[1], jnp.array([5., 13., 13., 8.]))

    c = [jnp.array([True, False, True, False])]
    a = [jnp.array([1., 2., 3., 4.])]
    b = [jnp.array(42.)]
    d = tree_util.tree_where(c, a, b)
    self.assertAllClose(d[0], jnp.array([1., 42., 3., 42.]))

    c = jnp.asarray(False)
    a = [jnp.array([1., 2., 3., 4.]), jnp.array([5., 6., 7., 8.])]
    b = jnp.array(42.)
    d = tree_util.tree_where(c, a, b)
    self.assertAllClose(d[0], jnp.array([42., 42., 42., 42.]))
    self.assertAllClose(d[1], jnp.array([42., 42., 42., 42.]))

  def test_broadcast_pytrees(self):
    # No leaf pytrees, treedefs match.
    trees_in = ([jnp.asarray(1.), jnp.asarray(2.)],
                [jnp.asarray(3.), jnp.asarray(4.)])
    trees_out = tree_util.broadcast_pytrees(*trees_in)
    for l_in, l_out in zip(jtu.tree_leaves(trees_in),
                           jtu.tree_leaves(trees_out)):
      self.assertAllClose(l_in, l_out)

    # One leaf pytree. `a_out` should match the structure of `b_in` and `b_out`
    # by broadcasting `a_in`.
    a_in = jnp.asarray(1.)
    b_in = [jnp.asarray(2.), jnp.asarray(3.)]
    a_out, b_out = tree_util.broadcast_pytrees(a_in, b_in)
    self.assertAllClose(a_out[0], a_in)
    self.assertAllClose(a_out[1], a_in)
    self.assertAllClose(b_out[0], b_in[0])
    self.assertAllClose(b_out[1], b_in[1])

    # Treedefs do not match.
    a_in = [jnp.asarray(1.)]
    b_in = [jnp.asarray(2.), jnp.asarray(3.)]
    with self.assertRaises(ValueError):
      a_out, b_out = tree_util.broadcast_pytrees(a_in, b_in)

if __name__ == '__main__':
  absltest.main()
