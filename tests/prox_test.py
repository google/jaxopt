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

from jaxopt import prox
from jaxopt import tree_util

import numpy as onp


class ProxTest(jtu.JaxTestCase):

  # A scalar implementation for check purposes.
  def _prox_l1(self, x, alpha):
    if x >= alpha:
      return x - alpha
    elif x <= -alpha:
      return x + alpha
    else:
      return 0

  def test_prox_lasso(self):
    rng = onp.random.RandomState(0)

    # Check forward pass with array x and scalar alpha.
    x = rng.rand(20) * 2 - 1
    alpha = 0.5
    expected = jnp.array([self._prox_l1(x[i], alpha) for i in range(len(x))])
    got = prox.prox_lasso(x, alpha)
    self.assertArraysAllClose(expected, got)

    # Check computed Jacobian against manual Jacobian.
    jac = jax.jacobian(prox.prox_lasso)(x, alpha)
    jac_exact = onp.zeros_like(jac)
    for i in range(len(x)):
      if x[i] >= alpha:
        jac_exact[i, i] = 1
      elif x[i] <= -alpha:
        jac_exact[i, i] = 1
      else:
        jac_exact[i, i] = 0
    self.assertArraysAllClose(jac_exact, jac)

    # Check forward pass with array x and array alpha.
    alpha = rng.rand(20)
    expected = jnp.array([self._prox_l1(x[i], alpha[i]) for i in range(len(x))])
    got = prox.prox_lasso(x, alpha)
    self.assertArraysAllClose(expected, got)

    # Check forward pass with pytree x and pytree alpha.
    x = (rng.rand(20) * 2 - 1, rng.rand(20) * 2 - 1)
    alpha = (rng.rand(20), rng.rand(20))
    expected0 = [self._prox_l1(x[0][i], alpha[0][i]) for i in range(len(x[0]))]
    expected1 = [self._prox_l1(x[1][i], alpha[1][i]) for i in range(len(x[0]))]
    expected = (jnp.array(expected0), jnp.array(expected1))
    got = prox.prox_lasso(x, alpha)
    self.assertArraysAllClose(jnp.array(expected), jnp.array(got))

    # Check forward pass with pytree x and tuple-of-scalars alpha.
    alpha = (0.5, 0.2)
    expected0 = [self._prox_l1(x[0][i], alpha[0]) for i in range(len(x[0]))]
    expected1 = [self._prox_l1(x[1][i], alpha[1]) for i in range(len(x[0]))]
    expected = (jnp.array(expected0), jnp.array(expected1))
    got = prox.prox_lasso(x, alpha)
    self.assertArraysAllClose(jnp.array(expected), jnp.array(got))

  def _prox_enet(self, x, lam, gamma):
    return (1.0 / (1.0 + lam * gamma)) * self._prox_l1(x, lam)

  def test_prox_elastic_net(self):
    rng = onp.random.RandomState(0)

    # Check forward pass with array x and scalar alpha.
    x = rng.rand(20) * 2 - 1
    params = (0.5, 0.1)
    expected = jnp.array([self._prox_enet(x[i], *params)
                          for i in range(len(x))])
    got = prox.prox_elastic_net(x, params)
    self.assertArraysAllClose(expected, got)

    # Check forward pass with array x and array alpha.
    params = (rng.rand(20), rng.rand(20))
    expected = jnp.array([self._prox_enet(x[i], params[0][i], params[1][i])
                          for i in range(len(x))])
    got = prox.prox_elastic_net(x, params)
    self.assertArraysAllClose(expected, got)

    # Check forward pass with pytree x.
    x = (rng.rand(20) * 2 - 1, rng.rand(20) * 2 - 1)
    params = (0.5, 0.1)
    expected0 = [self._prox_enet(x[0][i], *params) for i in range(len(x[0]))]
    expected1 = [self._prox_enet(x[1][i], *params) for i in range(len(x[0]))]
    expected = (jnp.array(expected0), jnp.array(expected1))
    got = prox.prox_elastic_net(x, ((0.5, 0.5), (0.1, 0.1)))
    self.assertArraysAllClose(jnp.array(expected), jnp.array(got))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
