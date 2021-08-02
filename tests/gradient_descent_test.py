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

from jaxopt import GradientDescent
from jaxopt import objective
from jaxopt._src import test_util

from sklearn import datasets
from sklearn import preprocessing


class GradientDescentTest(jtu.JaxTestCase):

  def test_logreg_with_intercept(self):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)
    data = (X, y)
    l2reg = 100.0
    fun = objective.l2_multiclass_logreg_with_intercept
    n_classes = len(jnp.unique(y))

    W_init = jnp.zeros((X.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    pytree_init = (W_init, b_init)
    gd = GradientDescent(fun=fun, tol=1e-3, maxiter=500)
    pytree_fit, info = gd.run(pytree_init, l2reg=l2reg, data=data)

    # Check optimality conditions.
    self.assertLessEqual(info.error, 5e-2)

    # Compare against sklearn.
    W_skl, b_skl = test_util.logreg_skl(X, y, l2reg, fit_intercept=True)
    self.assertArraysAllClose(pytree_fit[0], W_skl, atol=5e-2)
    self.assertArraysAllClose(pytree_fit[1], b_skl, atol=5e-2)

  def test_logreg_implicit_diff(self):
    X, y = datasets.load_digits(return_X_y=True)
    data = (X, y)
    l2reg = float(X.shape[0])
    fun = objective.l2_multiclass_logreg

    jac_num = test_util.logreg_skl_jac(X, y, l2reg)
    W_skl = test_util.logreg_skl(X, y, l2reg)

    # Make sure the decorator works.
    gd = GradientDescent(fun=fun, tol=1e-3, maxiter=10, implicit_diff=True)
    def wrapper(hyperparams):
      return gd.run(W_skl, hyperparams, data).params
    jac_custom = jax.jacrev(wrapper)(l2reg)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-2)

  @parameterized.product(acceleration=[True, False])
  def test_logreg_unrolled_autodiff(self, acceleration):
    X, y = datasets.load_digits(return_X_y=True)
    data = (X, y)
    l2reg = float(X.shape[0])
    fun = objective.l2_multiclass_logreg
    n_classes = len(jnp.unique(y))

    jac_l2reg = test_util.logreg_skl_jac(X, y, l2reg)

    # Compute the Jacobian w.r.t. l2reg via forward-mode autodiff.
    W_init = jnp.zeros((X.shape[1], n_classes))
    gd = GradientDescent(fun=fun, tol=1e-3, maxiter=200, implicit_diff=False,
                         acceleration=acceleration)
    def wrapper(hyperparams):
      return gd.run(W_init, hyperparams, data).params
    jac_l2reg2 = jax.jacfwd(wrapper)(l2reg)
    self.assertArraysAllClose(jac_l2reg, jac_l2reg2, atol=1e-2)

  def test_jit_and_vmap(self):
    X, y = datasets.make_classification(n_samples=30, n_features=5,
                                        n_informative=3, n_classes=2,
                                        random_state=0)
    data = (X, y)
    fun = objective.l2_multiclass_logreg
    hyperparams_list = jnp.array([1.0, 10.0])
    W_init = jnp.zeros((X.shape[1], 2))
    gd = GradientDescent(fun=fun, tol=1e-3, maxiter=100)

    def solve(hyperparams):
      W_fit, info = gd.run(W_init, hyperparams, data)
      return info.error

    errors = jnp.array([solve(hyperparams) for hyperparams in hyperparams_list])
    errors_vmap = jax.vmap(solve)(hyperparams_list)
    self.assertArraysAllClose(errors, errors_vmap, atol=1e-3)

    error0 = jax.jit(solve)(hyperparams_list[0])
    self.assertAllClose(errors[0], error0)

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  #jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
