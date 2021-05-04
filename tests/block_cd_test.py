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

import functools

import jax
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import block_cd
from jaxopt import implicit_diff
from jaxopt import projection
from jaxopt import prox
from jaxopt import test_util

import numpy as onp

from sklearn import datasets
from sklearn import preprocessing


class BlockCoordinateDescentTest(jtu.JaxTestCase):

  def _test_reg(self, block_prox, params_prox, maxiter, tol, atol, skl_fun,
                skl_fun_jac=None):
    X, y = datasets.make_regression(n_samples=50, n_features=10, random_state=0)
    fun = test_util.make_least_squares_objective(X, y)
    params_fun = 1.0

    # Check optimality conditions.
    w_init = jnp.zeros(X.shape[1])
    solver_fun = block_cd.make_solver_fun(fun=fun, init=w_init,
                                          block_prox=block_prox,
                                          maxiter=maxiter, tol=tol)
    w_fit = solver_fun(params_fun=params_fun, params_prox=params_prox)
    w_fit2 = block_prox(w_fit - jax.grad(fun)(w_fit, params_fun), params_prox)
    self.assertLessEqual(jnp.sqrt(jnp.sum((w_fit - w_fit2)**2)), tol * 1.1)

    # Compare against sklearn.
    w_skl = skl_fun(X, y, params_prox)
    self.assertArraysAllClose(w_fit, w_skl, atol=atol)

    # Check differentiation.
    if skl_fun_jac:
      jac_num = skl_fun_jac(X, y, params_prox)

      # By implicit diff.
      jac = jax.jacrev(solver_fun, argnums=1)(params_fun, params_prox)
      self.assertAllClose(jac_num, jac, atol=1e-1)

      # By autodiff.
      solver_fun = block_cd.make_solver_fun(fun=fun, init=w_init,
                                            block_prox=block_prox,
                                            implicit_diff=False,
                                            maxiter=maxiter, tol=tol)
      jac = jax.jacfwd(solver_fun, argnums=1)(params_fun, params_prox)
      self.assertAllClose(jac_num, jac, atol=1e-1)

  def test_lasso(self):
    self._test_reg(block_prox=prox.prox_lasso, params_prox=10.0, maxiter=1000,
                   tol=1e-3, atol=1e-2, skl_fun=test_util.lasso_skl,
                   skl_fun_jac=test_util.lasso_skl_jac)

  def test_elastic_net(self):
    self._test_reg(block_prox=prox.prox_elastic_net, params_prox=(2.0, 0.8),
                   maxiter=500, tol=1e-3, atol=1e-2, skl_fun=test_util.enet_skl,
                   skl_fun_jac=test_util.enet_skl_jac)

  def _test_multitask_reg(self, block_prox, params_prox, maxiter, tol, atol,
                          skl_fun):
    rng = onp.random.RandomState(0)
    n_samples, n_features, n_tasks = 50, 10, 3
    X = rng.randn(n_samples, n_features)
    W = rng.randn(n_features, n_tasks)
    Y = jnp.dot(X, W) + rng.randn(n_samples, n_tasks)

    fun = test_util.make_least_squares_objective(X, Y)
    prox_vmap = jax.vmap(block_prox, in_axes=(0, None))

    # Check optimality conditions.
    W_init = jnp.zeros((n_features, n_tasks))
    solver_fun = block_cd.make_solver_fun(fun=fun, init=W_init,
                                          block_prox=block_prox,
                                          maxiter=maxiter, tol=tol, verbose=0)
    W_fit = solver_fun(params_fun=1.0, params_prox=params_prox)
    W_fit2 = prox_vmap(W_fit - jax.grad(fun)(W_fit, 1.0), params_prox)
    self.assertLessEqual(jnp.sqrt(jnp.sum((W_fit - W_fit2)**2)), tol)

    # Compare against sklearn.
    W_skl = skl_fun(X, Y, params_prox, tol=tol)
    self.assertArraysAllClose(W_fit, W_skl, atol=1e-2)

  def test_multitask_lasso(self):
    self._test_multitask_reg(block_prox=prox.prox_group_lasso, params_prox=0.5,
                             maxiter=500, tol=1e-3, atol=1e-2,
                                  skl_fun=test_util.multitask_lasso_skl)


  def test_multiclass_linear_svm(self):
    n_samples, n_classes = 20, 3
    X, y = datasets.make_classification(n_samples=n_samples, n_features=5,
                                        n_informative=3, n_classes=n_classes,
                                        random_state=0)
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    prox = lambda x, pp, stepsize=1.0: projection.projection_simplex(x)
    prox_vmap = jax.vmap(prox, in_axes=(0, None))
    fun = test_util.make_multiclass_linear_svm_objective(X, y)
    tol = 1e-3
    maxiter = 500
    lam = 1000.0

    # Check optimality conditions.
    beta_init = jnp.ones((n_samples, n_classes)) / n_classes
    solver_fun = block_cd.make_solver_fun(fun=fun, init=beta_init,
                                          block_prox=prox, verbose=0,
                                          tol=tol, maxiter=maxiter)
    beta_fit = solver_fun(params_fun=lam)
    beta_fit2 = prox_vmap(beta_fit - jax.grad(fun)(beta_fit, lam), None)
    self.assertLessEqual(jnp.sqrt(jnp.sum((beta_fit - beta_fit2)**2)), 0.4)

    # Compare against sklearn.
    W_skl = test_util.multiclass_linear_svm_skl(X, y, lam)
    W_fit = jnp.dot(X.T, (Y - beta_fit)) / lam
    self.assertArraysAllClose(W_fit, W_skl, atol=1e-3)

  @parameterized.product(multiclass=[True, False], penalty=["l1", "l2"])
  def test_l1_logreg(self, multiclass, penalty):
    if multiclass:
      n_samples, n_features, n_classes = 20, 5, 3
      W_init = jnp.zeros((n_features, n_classes))
      make_objective_fun = test_util.make_logreg_objective
    else:
      n_samples, n_features, n_classes = 20, 5, 2
      W_init = jnp.zeros(n_features)
      make_objective_fun = test_util.make_binary_logreg_objective

    if penalty == "l1":
      block_prox = prox.prox_lasso
    else:
      block_prox = prox.prox_ridge

    X, y = datasets.make_classification(n_samples=n_samples,
                                        n_features=n_features,
                                        n_informative=3, n_classes=n_classes,
                                        random_state=0)
    fun = make_objective_fun(X, y, l2_penalty=False)
    tol = 1e-3
    maxiter = 500
    params_prox = 1.0
    params_fun = 1.0

    # Check optimality conditions.
    solver_fun = block_cd.make_solver_fun(fun=fun, init=W_init,
                                          block_prox=block_prox, verbose=0,
                                          tol=tol, maxiter=maxiter)
    W_fit = solver_fun(params_fun=params_fun, params_prox=params_prox)
    W_fit2 = block_prox(W_fit - jax.grad(fun)(W_fit, params_fun), params_prox)
    self.assertLessEqual(jnp.sqrt(jnp.sum((W_fit - W_fit2)**2)), tol)

    if not (multiclass and penalty == "l1"):

      # Compare against sklearn (it does not support multiclass + l1).
      W_skl = test_util.logreg_skl(X, y, params_prox, penalty=penalty,
                                   multiclass=multiclass)
      self.assertArraysAllClose(W_fit, W_skl, atol=1e-3)

      # Check differentiation.
      jac_num = test_util.logreg_skl_jac(X, y, params_prox, penalty=penalty,
                                         multiclass=multiclass)

      # By implicit diff.
      jac = jax.jacrev(solver_fun, argnums=1)(params_fun, params_prox)
      self.assertAllClose(jac_num, jac, atol=1e-1)

      # By autodiff.
      solver_fun = block_cd.make_solver_fun(fun=fun, init=W_init,
                                            block_prox=block_prox,
                                            implicit_diff=False,
                                            tol=tol, maxiter=maxiter)
      jac = jax.jacfwd(solver_fun, argnums=1)(params_fun, params_prox)
      self.assertAllClose(jac_num, jac, atol=1e-1)

  def test_vmap(self):
    make_solver_fun = functools.partial(block_cd.make_solver_fun,
                                        block_prox=prox.prox_lasso)
    make_fixed_point_fun = functools.partial(
        implicit_diff.make_block_cd_fixed_point_fun,
        block_prox=prox.prox_lasso)
    # A list of (params_fun, params_prox) pairs.
    params_list = jnp.array([[1.0, 1.0], [1.0, 10.0]])
    errors, errors_vmap = test_util.test_logreg_vmap(make_solver_fun,
                                                     make_fixed_point_fun,
                                                     params_list,
                                                     l2_penalty=False,
                                                     unpack_params=True)
    self.assertArraysAllClose(errors, errors_vmap, atol=1e-4)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
