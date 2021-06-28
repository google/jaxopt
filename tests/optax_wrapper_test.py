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

from jaxopt import optax_wrapper
from jaxopt import test_util2 as test_util

import numpy as onp

import optax

from sklearn import datasets


class OptaxWrapperTest(jtu.JaxTestCase):

  def test_logreg_with_intercept_manual_loop(self):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)
    data = (X, y)
    hyperparams = 100.0
    fun = test_util.l2_logreg_objective_with_intercept
    n_classes = len(jnp.unique(y))

    W_init = jnp.zeros((X.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    pytree_init = (W_init, b_init)

    opt = optax_wrapper.OptaxSolver(opt=optax.adam(1e-3), fun=fun)

    params, state = opt.init(pytree_init)
    for _ in range(200):
      params, state = opt.update(params, state, hyperparams, data)

    # Check optimality conditions.
    error = opt.l2_optimality_error(params, hyperparams, data)
    self.assertLessEqual(error, 0.01)

    # Compare against sklearn.
    W_skl, b_skl = test_util.logreg_skl(X, y, hyperparams, fit_intercept=True)
    self.assertArraysAllClose(params[0], W_skl, atol=5e-2)
    self.assertArraysAllClose(params[1], b_skl, atol=5e-2)

  def test_logreg_with_intercept_run(self):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)
    data = (X, y)
    hyperparams = 100.0
    fun = test_util.l2_logreg_objective_with_intercept
    n_classes = len(jnp.unique(y))

    W_init = jnp.zeros((X.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    pytree_init = (W_init, b_init)

    opt = optax_wrapper.OptaxSolver(opt=optax.adam(1e-3), fun=fun, maxiter=200)
    params, _ = opt.run(hyperparams, data, pytree_init)

    # Check optimality conditions.
    error = opt.l2_optimality_error(params, hyperparams, data)
    self.assertLessEqual(error, 0.01)

    # Compare against sklearn.
    W_skl, b_skl = test_util.logreg_skl(X, y, hyperparams, fit_intercept=True)
    self.assertArraysAllClose(params[0], W_skl, atol=5e-2)
    self.assertArraysAllClose(params[1], b_skl, atol=5e-2)

  def test_logreg_with_intercept_run_iterable(self):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)

    def dataset_loader(X, y, n_iter):
      rng = onp.random.RandomState(0)
      for _ in range(n_iter):
        perm = rng.permutation(len(X))
        yield X[perm], y[perm]

    hyperparams = 100.0
    fun = test_util.l2_logreg_objective_with_intercept
    n_classes = len(jnp.unique(y))

    W_init = jnp.zeros((X.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    pytree_init = (W_init, b_init)

    opt = optax_wrapper.OptaxSolver(opt=optax.adam(1e-3), fun=fun, maxiter=1000)
    iterable = dataset_loader(X, y, n_iter=200)
    params, _ = opt.run_iterator(hyperparams, iterable, pytree_init)

    # Check optimality conditions.
    error = opt.l2_optimality_error(params, hyperparams, (X, y))
    self.assertLessEqual(error, 0.01)

  def test_logreg_implicit_diff(self):
    X, y = datasets.load_digits(return_X_y=True)
    data = (X, y)
    lam = float(X.shape[0])
    fun = test_util.l2_logreg_objective

    jac_num = test_util.logreg_skl_jac(X, y, lam)
    W_skl = test_util.logreg_skl(X, y, lam)

    # Make sure the decorator works.
    opt = optax_wrapper.OptaxSolver(opt=optax.adam(1e-3), fun=fun, maxiter=5)
    jac_custom = jax.jacrev(test_util.first(opt.run))(lam, data, W_skl)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-2)

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
