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

from jax import test_util as jtu
import jax.numpy as jnp
from jaxopt import tree_util

import numpy as onp


class TreeUtilTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    rng = onp.random.RandomState(0)

    self.tree_A = (rng.randn(20, 10), rng.randn(20))
    self.tree_B = (rng.randn(20, 10), rng.randn(20))

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

    expected = (jnp.vdot(self.tree_A[0], self.tree_B[0]) +
                jnp.vdot(self.tree_A[1], self.tree_B[1]))
    got = tree_util.tree_vdot(self.tree_A, self.tree_B)
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


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
