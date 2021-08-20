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
from absl.testing import parameterized
import jax
from jax import test_util as jtu
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


class ProximalGradientTest(jtu.JaxTestCase):

  @parameterized.product(acceleration=[True, False])
  def test_lasso_manual_loop(self, acceleration):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)
    fun = objective.least_squares  # fun(params, data)
    lam = 10.0
    data = (X, y)

    pg = ProximalGradient(fun=fun, prox=prox.prox_lasso,
                          acceleration=acceleration)
    w_init = jnp.zeros(X.shape[1])
    params, state = pg.init(w_init, hyperparams_prox=lam, data=data)
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

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
