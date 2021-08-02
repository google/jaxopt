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
from absl.testing import parameterized

import jax
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import BlockCoordinateDescent
from jaxopt import objective
from jaxopt import projection
from jaxopt import prox
from jaxopt._src import test_util

import numpy as onp

from sklearn import datasets
from sklearn import preprocessing


class BlockCoordinateDescentTest(jtu.JaxTestCase):

  def test_lasso_manual_loop(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)

    # Setup parameters.
    fun = objective.least_squares  # fun(params, data)
    l2reg = 10.0
    data = (X, y)

    # Initialize.
    w_init = jnp.zeros(X.shape[1])
    bcd = BlockCoordinateDescent(fun=fun, block_prox=prox.prox_lasso)
    params, state = bcd.init(init_params=w_init, data=data)
    # Optimization loop.
    for _ in range(30):
      params, state = bcd.update(params=params, state=state,
                                 hyperparams_prox=l2reg, data=data)

    # Check optimality conditions.
    self.assertLess(state.error, 0.5)

  def test_lasso(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)

    # Set up parameters.
    fun = objective.least_squares  # fun(params, data)
    l2reg = 10.0
    data = (X, y)
    w_init = jnp.zeros(X.shape[1])

    # Run solver.
    bcd = BlockCoordinateDescent(fun=fun,
                                 block_prox=prox.prox_lasso,
                                 maxiter=150)
    sol = bcd.run(init_params=w_init, hyperparams_prox=l2reg, data=data)

    # Check optimality conditions.
    self.assertLess(sol.state.error, 0.01)

    # Check against sklearn.
    w_skl = test_util.lasso_skl(X, y, l2reg)
    self.assertArraysAllClose(sol.params, w_skl, atol=1e-2)

  def test_elastic_net(self):
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)

    # Set up parameters.
    fun = objective.least_squares  # fun(params, data)
    hyperparams_prox = (2.0, 0.8)
    data = (X, y)
    w_init = jnp.zeros(X.shape[1])

    # Run solver.
    bcd = BlockCoordinateDescent(fun=fun,
                                 block_prox=prox.prox_elastic_net,
                                 maxiter=100)
    sol = bcd.run(init_params=w_init,
                  hyperparams_prox=hyperparams_prox,
                  data=data)

    # Check optimality conditions.
    self.assertLess(sol.state.error, 0.01)

    # Check against sklearn.
    w_skl = test_util.enet_skl(X, y, hyperparams_prox)
    self.assertArraysAllClose(sol.params, w_skl, atol=1e-2)

  def test_multitask_reg(self):
    # Generate data.
    rng = onp.random.RandomState(0)
    n_samples, n_features, n_tasks = 50, 10, 3
    X = rng.randn(n_samples, n_features)
    W = rng.randn(n_features, n_tasks)
    Y = jnp.dot(X, W) + rng.randn(n_samples, n_tasks)

    # Set up parameters.
    fun = objective.least_squares  # fun(params, data)
    block_prox = prox.prox_group_lasso
    l2reg = 1e-1
    W_init = jnp.zeros((n_features, n_tasks))
    data = (X, Y)

    # Run solver.
    bcd = BlockCoordinateDescent(fun=fun, block_prox=block_prox,
                                 maxiter=1000, tol=1e-3)
    sol = bcd.run(init_params=W_init, hyperparams_prox=l2reg, data=data)

    # Check optimality conditions.
    self.assertLess(sol.state.error, 0.01)

    # Compare against sklearn.
    W_skl = test_util.multitask_lasso_skl(X, Y, l2reg * n_tasks)
    self.assertArraysAllClose(sol.params, W_skl, atol=1e-1)

  @parameterized.product(multiclass=[True, False], penalty=["l1", "l2"])
  def test_logreg(self, multiclass, penalty):
    # Generate data.
    if multiclass:
      n_samples, n_features, n_classes = 20, 5, 3
      W_init = jnp.zeros((n_features, n_classes))
    else:
      n_samples, n_features, n_classes = 20, 5, 2
      W_init = jnp.zeros(n_features)

    X, y = datasets.make_classification(n_samples=n_samples,
                                        n_features=n_features,
                                        n_informative=3, n_classes=n_classes,
                                        random_state=0)
    data = (X, y)

    # Set up parameters.
    if penalty == "l1":
      block_prox = prox.prox_lasso
    else:
      block_prox = prox.prox_ridge

    if multiclass:
      fun = objective.multiclass_logreg
    else:
      fun = objective.binary_logreg

    l2reg = 1e-2

    # Run solver.
    bcd = BlockCoordinateDescent(fun=fun, block_prox=block_prox,
                                 maxiter=3500, tol=1e-5)
    sol = bcd.run(W_init, hyperparams_prox=l2reg, data=data)

    # Check optimality conditions.
    self.assertLess(sol.state.error, 0.01)

    if not (multiclass and penalty == "l1"):

      # Compare against sklearn (it does not support multiclass + l1).
      W_skl = test_util.logreg_skl(X, y, l2reg, penalty=penalty,
                                   multiclass=multiclass)
      self.assertArraysAllClose(sol.params, W_skl, atol=1e-2)

      # Check differentiation.
      jac_num = test_util.logreg_skl_jac(X, y, l2reg, eps=1e-4,
                                         penalty=penalty, multiclass=multiclass)

      # By autodiff.
      bcd = BlockCoordinateDescent(fun=fun,
                                   block_prox=block_prox,
                                   maxiter=10000, tol=1e-6,
                                   implicit_diff=False)
      def wrapper(hyperparams_prox):
        return bcd.run(init_params=W_init,
                       hyperparams_prox=hyperparams_prox,
                       data=data).params
      jac = jax.jacfwd(wrapper)(l2reg)
      self.assertAllClose(jac_num, jac, atol=5e-1)

      # By implicit diff.
      bcd = BlockCoordinateDescent(fun=fun, block_prox=block_prox,
                                   maxiter=3500, tol=1e-5, implicit_diff=True)
      def wrapper(hyperparams_prox):
        return bcd.run(W_skl, hyperparams_prox, data).params
      jac = jax.jacrev(wrapper)(l2reg)
      self.assertAllClose(jac_num, jac, atol=1e-1)

  def test_multiclass_linear_svm(self):
    # Generate data.
    n_samples, n_classes = 20, 3
    X, y = datasets.make_classification(n_samples=n_samples, n_features=5,
                                        n_informative=3, n_classes=n_classes,
                                        random_state=0)
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    Y = jnp.array(Y)

    # Set up parameters.
    block_prox = prox.make_prox_from_projection(projection.projection_simplex)
    fun = objective.multiclass_linear_svm_dual
    data = (X, Y)
    l2reg = 1000.0
    beta_init = jnp.ones((n_samples, n_classes)) / n_classes

    # Run solver.
    bcd = BlockCoordinateDescent(fun=fun, block_prox=block_prox,
                                 maxiter=3500, tol=1e-5)
    sol = bcd.run(beta_init, hyperparams_prox=None, l2reg=l2reg, data=data)

    # Check optimality conditions.
    self.assertLess(sol.state.error, 0.01)

    # Compare against sklearn.
    W_skl = test_util.multiclass_linear_svm_skl(X, y, l2reg)
    W_fit = jnp.dot(X.T, (Y - sol.params)) / l2reg
    self.assertArraysAllClose(W_fit, W_skl, atol=1e-3)

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
