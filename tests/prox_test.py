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
import jax.numpy as jnp

from jaxopt import projection
from jaxopt import prox
from jaxopt._src import test_util

import numpy as onp


class ProxTest(test_util.JaxoptTestCase):

  def test_prox_none(self):
    rng = onp.random.RandomState(0)
    x = rng.rand(20) * 2 - 1
    self.assertArraysAllClose(prox.prox_none(x), x)

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
    hyperparams = (0.5, 0.1)
    expected = jnp.array([self._prox_enet(x[i], *hyperparams)
                          for i in range(len(x))])
    got = prox.prox_elastic_net(x, hyperparams)
    self.assertArraysAllClose(expected, got)

    # Check forward pass with array x and array alpha.
    hyperparams = (rng.rand(20), rng.rand(20))
    expected = jnp.array([self._prox_enet(x[i], hyperparams[0][i],
                                          hyperparams[1][i])
                          for i in range(len(x))])
    got = prox.prox_elastic_net(x, hyperparams)
    self.assertArraysAllClose(expected, got)

    # Check forward pass with pytree x.
    x = (rng.rand(20) * 2 - 1, rng.rand(20) * 2 - 1)
    hyperparams = (0.5, 0.1)
    expected0 = [self._prox_enet(x[0][i], *hyperparams)
                 for i in range(len(x[0]))]
    expected1 = [self._prox_enet(x[1][i], *hyperparams)
                 for i in range(len(x[0]))]
    expected = (jnp.array(expected0), jnp.array(expected1))
    got = prox.prox_elastic_net(x, ((0.5, 0.5), (0.1, 0.1)))
    self.assertArraysAllClose(jnp.array(expected), jnp.array(got))

  # A pure NumPy implementation for check purposes.
  def _prox_l2(self, x, alpha):
    l2_norm = onp.sqrt(onp.sum(x ** 2))
    return max(1 - alpha / l2_norm, 0) * x

  def test_prox_group_lasso(self):
    rng = onp.random.RandomState(0)
    x = rng.rand(20) * 2 - 1

    # An example with non-zero block.
    alpha = 0.1
    got = prox.prox_group_lasso(x, alpha)
    expected = self._prox_l2(x, alpha)

    # An example with zero block.
    alpha = 10.0
    self.assertArraysAllClose(got, expected)
    got = prox.prox_group_lasso(x, alpha)
    expected = self._prox_l2(x, alpha)

  def test_prox_ridge(self):
    rng = onp.random.RandomState(0)
    x = rng.rand(20) * 2 - 1
    x = jnp.array(x)
    alpha = 10.0

    # The objective solved by the prox is differentiable.
    # We can check that the gradient is zero.
    def fun(y):
      """f(y) = 0.5 ||y - x||^2 + 0.5 * alpha ||y||_2^2."""
      diff = x - y
      return 0.5 * jnp.sum(diff ** 2) + 0.5 * alpha * jnp.sum(y ** 2)

    got = prox.prox_ridge(x, alpha)
    self.assertArraysAllClose(jax.grad(fun)(got), jnp.zeros_like(got))

  def test_prox_non_negative_ridge(self):
    rng = onp.random.RandomState(0)
    x = rng.rand(20) * 2 - 1
    x = jnp.array(x)
    alpha = 10.0

    def fun(y):
      """f(y) = 0.5 ||y - x||^2 + 0.5 * alpha ||y||_2^2."""
      diff = x - y
      return 0.5 * jnp.sum(diff ** 2) + 0.5 * alpha * jnp.sum(y ** 2)

    got = prox.prox_non_negative_ridge(x, alpha)
    fixed_point = jax.nn.relu(got - jax.grad(fun)(got))
    self.assertArraysAllClose(got, fixed_point)

  def test_prox_non_negative_lasso(self):
    rng = onp.random.RandomState(0)
    x = rng.rand(20) * 2 - 1
    x = jnp.array(x)
    alpha = 0.5

    def fun(y):
      """f(y) = 0.5 ||y - x||^2 + alpha * sum(y)."""
      diff = x - y
      return 0.5 * jnp.sum(diff ** 2) + alpha * jnp.sum(y)

    got = prox.prox_non_negative_lasso(x, alpha)
    fixed_point = jax.nn.relu(got - jax.grad(fun)(got))
    self.assertArraysAllClose(got, fixed_point)

  def test_make_prox_from_projection(self):
    rng = onp.random.RandomState(0)
    x = rng.rand(10)
    proxop = prox.make_prox_from_projection(projection.projection_simplex)
    self.assertArraysAllClose(proxop(x), projection.projection_simplex(x))

if __name__ == '__main__':
  absltest.main()
