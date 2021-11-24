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
"""Linear Operator tests."""

from absl.testing import absltest


import jax
from jax import test_util as jtu
import jax.numpy as jnp
import numpy as onp

from jaxopt._src.linear_operator import FunctionalLinearOperator


class LinearOperatorTest(jtu.JaxTestCase):

  def test_matvec_and_rmatvec(self):
    rng = onp.random.RandomState(0)
    A = rng.randn(5, 4)
    matvec = lambda A,x: jnp.dot(A, x)
    x = rng.randn(4)
    y = rng.randn(5)
    linop_A = FunctionalLinearOperator(matvec, A)
    mv_A, rmv_A = linop_A.matvec_and_rmatvec(x, y)
    self.assertArraysAllClose(mv_A, jnp.dot(A, x))
    self.assertArraysAllClose(rmv_A, jnp.dot(A.T, y))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
