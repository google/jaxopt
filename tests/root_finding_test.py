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


from jaxopt import projection
from jaxopt import root_finding


def _projection_simplex_bisect(x, s=1.0):
  def threshold(x):
    # tau = max(x) => tau >= x_i for all i
    #              => x_i - tau <= 0 for all i
    #              => maximum(x_i - tau, 0) = 0 for all i
    #              => fun(tau, x) = -s <= 0
    lower = jax.lax.stop_gradient(jnp.max(x))
    # tau' = min(x) => tau' <= x_i for all i
    #               => 0 <= x_i - tau' for all_i
    #               => maximum(x_i - tau', 0) >= 0
    #               => fun(tau, x) >= 0 where tau = tau' - s / len(x)
    upper = jax.lax.stop_gradient(jnp.min(x)) - s / len(x)
    # fun(tau, x) is a decreasing function of x on [lower, upper]
    # since the derivative w.r.t. tau is negative.
    fun = lambda tau, x_: jnp.sum(jnp.maximum(x_ - tau, 0)) - s
    bisect_fun = root_finding.make_bisect_fun(fun=fun, lower=lower, upper=upper,
                                              increasing=False)
    return bisect_fun(x)
  return jnp.maximum(x - threshold(x), 0)


class RootFindingTest(jtu.JaxTestCase):

  def test_bisect(self):
    rng = onp.random.RandomState(0)

    for _ in range(10):
      x = jnp.array(rng.randn(50).astype(onp.float32))
      p = projection.projection_simplex(x)
      p2 = _projection_simplex_bisect(x)
      self.assertArraysAllClose(p, p2, atol=1e-4)

      J = jax.jacrev(projection.projection_simplex)(x)
      J2 = jax.jacrev(_projection_simplex_bisect)(x)
      self.assertArraysAllClose(J, J2, atol=1e-5)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
