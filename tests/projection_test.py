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

import numpy as onp


from jaxopt.projection import projection_simplex


class ProjectionTest(jtu.JaxTestCase):

  def test_projection_simplex(self):
    rng = onp.random.RandomState(0)

    for _ in range(10):
      x = rng.randn(50).astype(onp.float32)

      p = projection_simplex(x)
      self.assertAllClose(jnp.sum(p), 1.0)
      self.assertTrue(jnp.all(0 <= p))
      self.assertTrue(jnp.all(p <= 1))

      p = projection_simplex(x, 0.8)
      self.assertAllClose(jnp.sum(p), 0.8)
      self.assertTrue(jnp.all(0 <= p))
      self.assertTrue(jnp.all(p <= 0.8))

  def test_projection_simplex_jacobian(self):
    rng = onp.random.RandomState(0)
    x = rng.rand(5).astype(onp.float32)
    v = rng.randn(5).astype(onp.float32)
    p = projection_simplex(x)

    J = jax.jacrev(projection_simplex)(x)

    # Check against theoretical expression.
    support = (p > 0).astype(jnp.int32)
    cardinality = jnp.count_nonzero(support)
    J_true = jnp.diag(support) - jnp.outer(support, support) / cardinality
    self.assertArraysAllClose(J_true, J)

    # Check that Jacobian is symmetric.
    self.assertArraysAllClose(J, J.T)

    # Check vector Jacobian product.
    vJ, = jax.vjp(projection_simplex, x)[1](v)
    self.assertArraysAllClose(vJ, jnp.dot(v, J_true))

    # Check vector Jacobian product.
    Jv = jax.jvp(projection_simplex, (x,), (v,))[1]
    self.assertArraysAllClose(Jv, jnp.dot(J_true,v))

  def test_projection_simplex_vmap(self):
    rng = onp.random.RandomState(0)
    X = rng.randn(3, 50).astype(onp.float32)

    # Check with default s=1.0.
    P = jax.vmap(projection_simplex)(X)
    self.assertArraysAllClose(jnp.sum(P, axis=1), jnp.ones(len(X)))
    self.assertTrue(jnp.all(0 <= P))
    self.assertTrue(jnp.all(P <= 1))

    # Check with s=0.8.
    P = jax.vmap(projection_simplex)(X, 0.8 * jnp.ones(len(X)))
    self.assertArraysAllClose(jnp.sum(P, axis=1), 0.8 * jnp.ones(len(X)))
    self.assertTrue(jnp.all(0 <= P))
    self.assertTrue(jnp.all(P <= 0.8))

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
