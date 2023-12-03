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
import jax.numpy as jnp

from jaxopt import objective
from jaxopt import PolyakSGD
from jaxopt._src import test_util

import numpy as onp
from sklearn import datasets


class PolyakSgdTest(test_util.JaxoptTestCase):

  @parameterized.product(momentum=[0.0, 0.9], sps_variant=['SPS_max', 'SPS+'])
  def test_logreg_overparameterized(self, momentum, sps_variant):
    # Test SPS on an over-parameterized logistic regression problem.
    # The loss' infimum is zero and SPS should converge to a minimizer.
    data = datasets.make_classification(
        n_samples=10, n_features=10, random_state=0
    )
    # fun(params, data)
    fun = objective.l2_multiclass_logreg_with_intercept
    n_classes = len(jnp.unique(data[1]))

    w_init = jnp.zeros((data[0].shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    params = (w_init, b_init)

    opt = PolyakSGD(fun=fun, fun_min=0, momentum=momentum, variant=sps_variant)
    error_init = opt.l2_optimality_error(params, l2reg=0, data=data)
    params, _ = opt.run(params, l2reg=0., data=data)

    # Check optimality conditions.
    error = opt.l2_optimality_error(params, l2reg=0., data=data)
    self.assertLessEqual(error / error_init, 0.01)

  @parameterized.product(momentum=[0.0, 0.9], sps_variant=['SPS_max', 'SPS+'])
  def test_logreg_with_intercept_manual_loop(self, momentum, sps_variant):
    x, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)
    data = (x, y)
    l2reg = 0.1
    # fun(params, l2reg, data)
    fun = objective.l2_multiclass_logreg_with_intercept
    n_classes = len(jnp.unique(y))

    w_init = jnp.zeros((x.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    params = (w_init, b_init)

    opt = PolyakSGD(
        fun=fun, fun_min=0.6975, momentum=momentum, variant=sps_variant
    )
    error_init = opt.l2_optimality_error(params, l2reg=l2reg, data=data)

    state = opt.init_state(params, l2reg=l2reg, data=data)
    for _ in range(200):
      params, state = opt.update(params, state, l2reg=l2reg, data=data)

    # Check optimality conditions.
    error = opt.l2_optimality_error(params, l2reg=l2reg, data=data)
    self.assertLessEqual(error / error_init, 0.02)

  @parameterized.product(has_aux=[True, False])
  def test_logreg_with_intercept_run(self, has_aux):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)
    data = (X, y)
    l2reg = 100.0
    _fun = objective.l2_multiclass_logreg_with_intercept
    if has_aux:
      def fun(params, l2reg, data):
        return _fun(params, l2reg, data), None
    else:
      fun = _fun

    n_classes = len(jnp.unique(y))

    W_init = jnp.zeros((X.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    pytree_init = (W_init, b_init)

    opt = PolyakSGD(fun=fun, max_stepsize=0.01, maxiter=300, has_aux=has_aux)
    # Test positional, keyword and mixed arguments.
    for params, _ in (opt.run(pytree_init, l2reg, data),
                      opt.run(pytree_init, l2reg=l2reg, data=data),
                      opt.run(pytree_init, l2reg, data=data)):

      # Check optimality conditions.
      error = opt.l2_optimality_error(params, l2reg=l2reg, data=data)
      self.assertLessEqual(error, 0.05)

  def test_logreg_with_intercept_run_iterable(self):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)

    def dataset_loader(X, y, n_iter):
      rng = onp.random.RandomState(0)
      for _ in range(n_iter):
        perm = rng.permutation(len(X))
        yield X[perm], y[perm]

    l2reg = 100.0
    fun = objective.l2_multiclass_logreg_with_intercept
    n_classes = len(jnp.unique(y))

    W_init = jnp.zeros((X.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    pytree_init = (W_init, b_init)

    opt = PolyakSGD(fun=fun, max_stepsize=0.01, maxiter=1000)
    iterable = dataset_loader(X, y, n_iter=200)
    params, _ = opt.run_iterator(pytree_init, iterable, l2reg=l2reg)

    # Check optimality conditions.
    error = opt.l2_optimality_error(params, l2reg=l2reg, data=(X, y))
    self.assertLessEqual(error, 0.05)

  def test_logreg_autodiff(self):
    X, y = datasets.load_digits(return_X_y=True)
    data = (X, y)
    l2reg = float(X.shape[0])
    fun = objective.l2_multiclass_logreg

    jac_num = test_util.logreg_skl_jac(X, y, l2reg)
    W_skl = test_util.logreg_skl(X, y, l2reg)

    # Make sure the decorator works.
    opt = PolyakSGD(fun=fun, max_stepsize=1e-3, maxiter=5)
    def wrapper(hyperparams):
      return opt.run(W_skl, l2reg=l2reg, data=data).params
    jac_custom = jax.jacrev(wrapper)(l2reg)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-2)

  def test_logreg_implicit_diff(self):
    X, y = datasets.load_digits(return_X_y=True)
    data = (X, y)
    l2reg = float(X.shape[0])
    fun = objective.l2_multiclass_logreg

    jac_num = test_util.logreg_skl_jac(X, y, l2reg)
    W_skl = test_util.logreg_skl(X, y, l2reg)

    # Make sure the decorator works.
    opt = PolyakSGD(fun=fun, max_stepsize=1e-3, maxiter=5, implicit_diff=True)
    def wrapper(hyperparams):
      # Unfortunately positional arguments are required when implicit_diff=True.
      return opt.run(W_skl, l2reg, data).params
    jac_custom = jax.jacrev(wrapper)(l2reg)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-2)

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  #jax.config.update("jax_enable_x64", True)
  absltest.main()
