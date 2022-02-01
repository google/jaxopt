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
from jaxopt import Bisection
from jaxopt._src import test_util

import numpy as onp


# optimality_fun(params, hyperparams, data)
def _optimality_fun_proj_simplex(tau, x, s):
  # optimality_fun(tau, x, s) is a decreasing function of tau on
  # [lower, upper] since the derivative w.r.t. tau is negative.
  return jnp.sum(jnp.maximum(x - tau, 0)) - s


def _threshold_proj_simplex(x, s):
  # tau = max(x) => tau >= x_i for all i
  #              => x_i - tau <= 0 for all i
  #              => maximum(x_i - tau, 0) = 0 for all i
  #              => optimality_fun(tau, x, s) = -s <= 0
  upper = jax.lax.stop_gradient(jnp.max(x))

  # tau' = min(x) => tau' <= x_i for all i
  #               => 0 <= x_i - tau' for all_i
  #               => maximum(x_i - tau', 0) >= 0
  #               => optimality_fun(tau, x, s) >= 0
  # where tau = tau' - s / len(x)
  lower = jax.lax.stop_gradient(jnp.min(x)) - s / len(x)

  bisect = Bisection(optimality_fun=_optimality_fun_proj_simplex,
                     lower=lower, upper=upper, check_bracket=False)
  return bisect.run(None, x, s).params


def _projection_simplex_bisect(x, s=1.0):
  return jnp.maximum(x - _threshold_proj_simplex(x, s), 0)


class BisectionTest(test_util.JaxoptTestCase):

  def test_bisect(self):
    rng = onp.random.RandomState(0)

    jitted_fun = jax.jit(_projection_simplex_bisect)

    for _ in range(10):
      x = jnp.array(rng.randn(50).astype(onp.float32))
      p = projection.projection_simplex(x)
      p2 = _projection_simplex_bisect(x)
      p3 = jitted_fun(x)
      self.assertArraysAllClose(p, p2, atol=1e-4)
      self.assertArraysAllClose(p, p3, atol=1e-4)

      J = jax.jacrev(projection.projection_simplex)(x)
      J2 = jax.jacrev(_projection_simplex_bisect)(x)
      J3 = jax.jacrev(jitted_fun)(x)
      self.assertArraysAllClose(J, J2, atol=1e-5)
      self.assertArraysAllClose(J, J3, atol=1e-5)

  def test_bisect_wrong_lower_bracket(self):
    rng = onp.random.RandomState(0)
    x = jnp.array(rng.randn(5).astype(onp.float32))
    s = 1.0
    upper = jnp.max(x)
    bisect = Bisection(optimality_fun=_optimality_fun_proj_simplex,
                       lower=upper, upper=upper)
    self.assertRaises(ValueError, bisect.run, None, x, s)

  def test_bisect_wrong_upper_bracket(self):
    rng = onp.random.RandomState(0)
    x = jnp.array(rng.randn(5).astype(onp.float32))
    s = 1.0
    lower = jnp.min(x) - s / len(x)
    bisect = Bisection(optimality_fun=_optimality_fun_proj_simplex,
                       lower=lower, upper=lower)
    self.assertRaises(ValueError, bisect.run, None, x, s)

  def test_grad_of_value_and_grad(self):
    # See https://github.com/google/jaxopt/issues/141

    bisect = lambda x: _projection_simplex_bisect(x)[0]

    def bisect_val(x):
      val, _ = jax.value_and_grad(bisect)(x)
      return val

    rng = onp.random.RandomState(0)
    x = jnp.array(rng.randn(5).astype(onp.float32))
    g1 = jax.grad(bisect)(x)
    g2 = jax.grad(bisect_val)(x)
    self.assertArraysAllClose(g1, g2)


if __name__ == '__main__':
  absltest.main()
