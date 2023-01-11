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
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from jaxopt import implicit_diff
from jaxopt import objective
from jaxopt import projection
from jaxopt import prox
from jaxopt import ProximalGradient
from jaxopt._src import test_util
from jaxopt import tree_util as tu

from sklearn import datasets
from sklearn import preprocessing


def make_stepsize_schedule(max_stepsize, n_steps, power=1.0) -> Callable:
  def stepsize_schedule(t: int) -> float:
    true_fn = lambda t: 1.0
    false_fn = lambda t: (n_steps / t) ** power
    decay_factor = jax.lax.cond(t <= n_steps, true_fn, false_fn, t)
    return decay_factor * max_stepsize
  return stepsize_schedule


N_CALLS = 0

class ProximalGradientTest(test_util.JaxoptTestCase):

  @parameterized.product(acceleration=[True, False])
  def test_lasso_manual_loop(self, acceleration):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    fun = objective.least_squares  # fun(params, data)
    lam = 10.0
    data = (X, y)

    pg = ProximalGradient(fun=fun, prox=prox.prox_lasso,
                          acceleration=acceleration)
    params = jnp.zeros(X.shape[1])
    state = pg.init_state(params, hyperparams_prox=lam, data=data)
    for _ in range(10):
      params, state = pg.update(params, state, hyperparams_prox=lam, data=data)

    # Check optimality conditions.
    self.assertLess(state.error, 2e-2)

  @parameterized.product(acceleration=[True, False])
  def test_lasso(self, acceleration=True):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    fun = objective.least_squares
    lam = 10.0
    data = (X, y)

    w_init = jnp.zeros(X.shape[1])
    pg = ProximalGradient(fun=fun, prox=prox.prox_lasso, maxiter=200, tol=1e-3,
                          acceleration=acceleration)
    w_fit, info = pg.run(w_init, hyperparams_prox=lam, data=data)

    # Check optimality conditions.
    self.assertLess(info.error, 1e-3)

    # Compare against sklearn.
    w_skl = test_util.lasso_skl(X, y, lam)
    self.assertArraysAllClose(w_fit, w_skl, atol=1e-2)

  def test_lasso_implicit_diff(self):
    """Test implicit differentiation of a single lambda parameter."""
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    lam = 10.0
    data = (X, y)

    fun = objective.least_squares
    jac_num = test_util.lasso_skl_jac(X, y, lam)
    w_skl = test_util.lasso_skl(X, y, lam)

    pg = ProximalGradient(fun=fun, prox=prox.prox_lasso, tol=1e-3, maxiter=200,
                          acceleration=True, implicit_diff=True)

    def wrapper(hyperparams_prox):
      return pg.run(w_skl, hyperparams_prox, data).params

    jac_prox = jax.jacrev(wrapper)(lam)
    self.assertArraysAllClose(jac_num, jac_prox, atol=1e-3)

  def test_lasso_implicit_diff_with_kwargs(self):
    """Same as above but with kwargs."""
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    lam = 10.0
    data = (X, y)

    fun = objective.least_squares
    jac_num = test_util.lasso_skl_jac(X, y, lam)
    w_skl = test_util.lasso_skl(X, y, lam)

    pg = ProximalGradient(fun=fun, prox=prox.prox_lasso, tol=1e-3, maxiter=200,
                          acceleration=True, implicit_diff=True)

    def wrapper(hyperparams_prox):
      return pg.run(w_skl, hyperparams_prox=hyperparams_prox, data=data).params

    jac_prox = jax.jacrev(wrapper)(lam)
    self.assertArraysAllClose(jac_num, jac_prox, atol=1e-3)

  def test_stepsize_schedule(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    fun = objective.least_squares  # fun(params, data)
    data = (X, y)

    schedule = make_stepsize_schedule(max_stepsize=1., n_steps=80, power=1.0)

    pg = ProximalGradient(fun=fun, stepsize=schedule,
                          prox=prox.prox_none,
                          acceleration=False)
    params = jnp.zeros(X.shape[1])
    _, state = pg.run(params, hyperparams_prox=None, data=data)
    self.assertLess(state.error, 1e-3)

  @parameterized.product(n_iter=[10])
  def test_n_calls(self, n_iter):
    """Test whether the number of function calls
    is equal to the number of iterations + 1 in the
    no linesearch case, where the complexity is linear."""
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    orig_fun = objective.least_squares
    def fun(params, data):
      global N_CALLS
      N_CALLS += 1
      return orig_fun(params, data)
    lam = 10.0
    data = (X, y)

    w_init = jnp.zeros(X.shape[1])
    pg = ProximalGradient(fun=fun, prox=prox.prox_lasso, maxiter=n_iter, tol=1e-10, stepsize=1e-5,
                          acceleration=True, jit=False)
    w_fit, info = pg.run(w_init, hyperparams_prox=lam, data=data)

    self.assertEqual(N_CALLS, n_iter)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main()
