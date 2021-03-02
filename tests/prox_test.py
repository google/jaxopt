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

import jax
from jax import test_util as jtu
import jax.numpy as jnp
from jaxopt.prox import prox_l1

import numpy as onp


class ProxTest(jtu.JaxTestCase):

  def test_prox_l1(self):
    # A trivial non-vectorized implementation for check purposes.
    def _prox_l1(x, alpha):
      if x >= alpha:
        return x - alpha
      elif x <= -alpha:
        return x + alpha
      else:
        return 0

    rng = onp.random.RandomState(0)
    x = rng.rand(20) * 2 - 1
    alpha = 0.5
    expected = jnp.array([_prox_l1(x[i], alpha) for i in range(len(x))])
    self.assertArraysAllClose(prox_l1(x, alpha), expected)

    jacobian1 = jax.jacobian(prox_l1)(x, alpha)

    # Compute Jacobian matrix manually.
    jacobian2 = onp.zeros_like(jacobian1)
    for i in range(len(x)):
      if x[i] >= alpha:
        jacobian2[i, i] = 1
      elif x[i] <= -alpha:
        jacobian2[i, i] = 1
      else:
        jacobian2[i, i] = 0

    # Check computed Jacobian against manual Jacobian.
    self.assertArraysAllClose(jacobian1, jacobian2)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
