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

import functools

from absl.testing import absltest

import jax
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import objective
from jaxopt import projection
from jaxopt import ProjectedGradient
from jaxopt import ScipyBoundedMinimize

import numpy as onp


class ProjectedGradientTest(jtu.JaxTestCase):

  def test_non_negative_least_squares(self):
    rng = onp.random.RandomState(0)
    X = rng.randn(10, 5)
    w = rng.rand(5)
    y = jnp.dot(X, w)
    fun = objective.least_squares
    w_init = jnp.zeros_like(w)

    pg = ProjectedGradient(fun=fun,
                           projection=projection.projection_non_negative)
    pg_sol = pg.run(w_init, data=(X, y)).params

    lbfgsb = ScipyBoundedMinimize(fun=fun, method="l-bfgs-b")
    lower_bounds = jnp.zeros_like(w_init)
    upper_bounds = jnp.ones_like(w_init) * jnp.inf
    bounds = (lower_bounds, upper_bounds)
    lbfgsb_sol = lbfgsb.run(w_init, bounds=bounds, data=(X, y)).params

    self.assertArraysAllClose(pg_sol, lbfgsb_sol, atol=1e-2)

  def test_projected_gradient_l2_ball(self):
    rng = onp.random.RandomState(0)
    X = rng.randn(10, 5)
    w = rng.rand(5)
    y = jnp.dot(X, w)
    fun = objective.least_squares
    w_init = jnp.zeros_like(w)

    pg = ProjectedGradient(fun=fun,
                           projection=projection.projection_l2_ball)
    pg_sol = pg.run(w_init, hyperparams_proj=1.0, data=(X, y)).params
    self.assertLess(jnp.sqrt(jnp.sum(pg_sol ** 2)), 1.0)

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
