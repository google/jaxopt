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
import jax.numpy as jnp

from jaxopt import objective
from jaxopt import projection
from jaxopt import ProjectedGradient
from jaxopt import ScipyBoundedMinimize
from jaxopt._src import test_util

import numpy as onp


class ProjectedGradientTest(test_util.JaxoptTestCase):

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

  def test_projected_gradient_l2_ball_manual_loop(self):
    rng = onp.random.RandomState(0)
    X = rng.randn(10, 5)
    w = rng.rand(5)
    y = jnp.dot(X, w)
    fun = objective.least_squares
    params = jnp.zeros_like(w)

    pg = ProjectedGradient(fun=fun,
                           projection=projection.projection_l2_ball)

    state = pg.init_state(params)

    for _ in range(10):
      params, state = pg.update(params, state, hyperparams_proj=1.0, data=(X, y))

    self.assertLess(jnp.sqrt(jnp.sum(params ** 2)), 1.0)

  def test_projected_gradient_implicit_diff(self):
    rng = onp.random.RandomState(0)
    X = rng.randn(10, 5)
    w = rng.rand(5)
    y = jnp.dot(X, w)
    fun = objective.least_squares
    w_init = jnp.zeros_like(w)

    def solution(radius):
      pg = ProjectedGradient(fun=fun,
                             projection=projection.projection_l2_ball)
      return pg.run(w_init, hyperparams_proj=radius, data=(X, y)).params

    eps = 1e-4
    J = jax.jacobian(solution)(0.1)
    J2 = (solution(0.1 + eps) - solution(0.1 - eps)) / (2 * eps)
    self.assertArraysAllClose(J, J2, atol=1e-2)

  def test_polyhedron_projection(self):
    def f(x):
      return x[0]**2-x[1]**2

    A = jnp.array([[0, 0]])
    b = jnp.array([0])
    G = jnp.array([[-1, -1], [0, 1], [1, -1], [-1, 0], [0, -1]])
    h = jnp.array([-1, 1, 1, 0, 0])
    hyperparams = (A, b, G, h)

    proj = projection.projection_polyhedron
    pg = ProjectedGradient(fun=f, projection=proj, jit=False)
    sol, state = pg.run(init_params=jnp.array([0.,1.]), hyperparams_proj=hyperparams)
    self.assertLess(state.error, pg.tol)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main()
