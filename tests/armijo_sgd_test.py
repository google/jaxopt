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

import itertools

import jax
import jax.numpy as jnp

from jaxopt import objective
from jaxopt import ArmijoSGD
from jaxopt._src import test_util

import numpy as onp
from sklearn import datasets


class ArmijoSgdTest(test_util.JaxoptTestCase):

  @parameterized.product(aggressiveness=[0.1, 0.5, 0.9], decrease_factor=[0.8, 0.9])
  def test_interpolating_regime(self, aggressiveness, decrease_factor):
    """Test Armijo on a problem on which interpolation holds.

    No regularization, one cluster per class, two classes, with enough distance,
    to ensure classes are easily separable.
    """
    X, y = datasets.make_classification(n_samples=100, n_features=20, n_classes=2,
                                        n_informative=10, n_redundant=10,
                                        n_clusters_per_class=1, class_sep=6.,
                                        random_state=42)
    data = (X, y)
    l2reg = 0.0
    # fun(params, data)
    fun = objective.l2_multiclass_logreg_with_intercept
    n_classes = len(jnp.unique(y))

    W_init = jnp.zeros((X.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    pytree_init = (W_init, b_init)

    tol = 1e-5
    opt = ArmijoSGD(fun=fun, aggressiveness=aggressiveness, decrease_factor=decrease_factor,
                    maxiter=20*1000, tol=tol)
    params, state = opt.run(pytree_init, l2reg=l2reg, data=data)
    loss = opt.fun(params, l2reg, data)
    self.assertLessEqual(state.error, tol)
    self.assertLessEqual(loss, tol)  # check interpolation
    error = opt.l2_optimality_error(params, l2reg=l2reg, data=data)
    self.assertLessEqual(error, tol)

  @parameterized.product(l2reg=[1e-1, 1])
  def test_non_interpolating_regime(self, l2reg):
    """Test Armijo on a problem on which interpolation does not holds.

    No (theoretical) convergence guarantees.
    """
    X, y = datasets.make_classification(n_samples=200, n_features=10, n_classes=3,
                                        n_informative=5, n_redundant=3,
                                        n_clusters_per_class=2, class_sep=1.,
                                        random_state=42)
    data = (X, y)
    # fun(params, data)
    fun = objective.l2_multiclass_logreg_with_intercept
    n_classes = len(jnp.unique(y))

    W_init = jnp.zeros((X.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    pytree_init = (W_init, b_init)

    tol = 1e-7
    opt = ArmijoSGD(fun=fun, aggressiveness=0.7, maxiter=10*1000, tol=tol)
    params, state = opt.run(pytree_init, l2reg=l2reg, data=data)
    error = opt.l2_optimality_error(params, l2reg=l2reg, data=data)
    self.assertLessEqual(error, 0.0015)

  def test_logreg_autodiff(self):
    X, y = datasets.load_digits(return_X_y=True)
    data = (X, y)
    l2reg = float(X.shape[0])
    fun = objective.l2_multiclass_logreg

    jac_num = test_util.logreg_skl_jac(X, y, l2reg)
    W_skl = test_util.logreg_skl(X, y, l2reg)

    # Make sure the decorator works.
    opt = ArmijoSGD(fun=fun, maxiter=5)
    def wrapper(l2reg):
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
    opt = ArmijoSGD(fun=fun, maxiter=5)
    def wrapper(l2reg):
      # Unfortunately positional arguments are required when implicit_diff=True.
      return opt.run(W_skl, l2reg, data).params
    jac_custom = jax.jacrev(wrapper)(l2reg)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-2)

  def test_goldstein(self):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)

    def dataset_loader(X, y, n_iter):
      rng = onp.random.RandomState(0)
      for _ in range(n_iter):
        perm = rng.permutation(len(X))
        yield X[perm], y[perm]

    l2reg = 10.0
    fun = objective.l2_multiclass_logreg_with_intercept
    n_classes = len(jnp.unique(y))

    W_init = jnp.zeros((X.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    params = (W_init, b_init)

    tol = 1e-3
    opt = ArmijoSGD(fun=fun, reset_option='goldstein', maxiter=1000, tol=tol)
    iterable = dataset_loader(X, y, n_iter=200)
    state = opt.init_state(params, l2reg=l2reg, data=(X, y))
    @jax.jit
    def jitted_update(params, state, data):
      return opt.update(params, state, l2reg=l2reg, data=data)
    for data in itertools.islice(iterable, 0, opt.maxiter):
      params, state = jitted_update(params, state, data)
    # Check optimality conditions.
    error = opt.l2_optimality_error(params, l2reg=l2reg, data=(X, y))
    self.assertLessEqual(error, tol)

  def test_run_iterable(self):
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

    tol = 3e-1
    # few iterations due to speed issues
    opt = ArmijoSGD(fun=fun, maxiter=10, tol=tol)
    iterable = dataset_loader(X, y, n_iter=200)
    params, _ = opt.run_iterator(pytree_init, iterable, l2reg=l2reg)

    # Check optimality conditions.
    error = opt.l2_optimality_error(params, l2reg=l2reg, data=(X, y))
    self.assertLessEqual(error, tol)

  @parameterized.product(momentum=[0.5, 0.9])
  def test_momentum(self, momentum):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)
    data = (X, y)
    l2reg = 100.0
    # fun(params, data)
    fun = objective.l2_multiclass_logreg_with_intercept
    n_classes = len(jnp.unique(y))

    W_init = jnp.zeros((X.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    pytree_init = (W_init, b_init)

    opt = ArmijoSGD(fun=fun, momentum=momentum)

    params, state = opt.run(pytree_init, l2reg=l2reg, data=data)

    # Check optimality conditions.
    error = opt.l2_optimality_error(params, l2reg=l2reg, data=data)
    self.assertLessEqual(error, 0.05)


  def test_logreg_with_intercept_run(self):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)
    data = (X, y)
    l2reg = 100.0
    _fun = objective.l2_multiclass_logreg_with_intercept
    def fun(params, l2reg, data):
      return _fun(params, l2reg, data), None

    n_classes = len(jnp.unique(y))

    W_init = jnp.zeros((X.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    pytree_init = (W_init, b_init)

    opt = ArmijoSGD(fun=fun, maxiter=300, has_aux=True)
    # Test positional, keyword and mixed arguments.
    for params, _ in (opt.run(pytree_init, l2reg, data),
                      opt.run(pytree_init, l2reg=l2reg, data=data),
                      opt.run(pytree_init, l2reg, data=data)):

      # Check optimality conditions.
      error = opt.l2_optimality_error(params, l2reg=l2reg, data=data)
      self.assertLessEqual(error, 0.05)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main()
