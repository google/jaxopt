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
from jax.config import config
from jax import test_util as jtu
from jax.tree_util import tree_map, tree_all
from jax.test_util import check_grads
import optax

from jaxopt.tree_util import tree_l2_norm, tree_scalar_mul, tree_sub
from jaxopt import objective

from jaxopt import projection
from jaxopt import prox
from jaxopt._src import test_util

from jaxopt import AndersonWrapper
from jaxopt import BlockCoordinateDescent
from jaxopt import GradientDescent
from jaxopt import OptaxSolver
from jaxopt import PolyakSGD
from jaxopt import ProximalGradient

import numpy as onp
import scipy
from sklearn import datasets


class AndersonWrapperTest(jtu.JaxTestCase):

  def test_proximal_gradient_wrapper(self):
    """Baseline test on simple optimizer."""
    X, y = datasets.make_regression(n_samples=100, n_features=20, random_state=0)
    fun = objective.least_squares
    lam = 10.0
    data = (X, y)
    w_init = jnp.zeros(X.shape[1])
    tol = 1e-3
    maxiter = 1000
    pg = ProximalGradient(fun=fun, prox=prox.prox_lasso, maxiter=maxiter, tol=tol,
                          acceleration=False)
    aw = AndersonWrapper(pg, history_size=15)
    aw_params, awpg_info = aw.run(w_init, hyperparams_prox=lam, data=data)
    self.assertLess(awpg_info.error, tol)

  def test_mixing_frequency_polyak(self):
    """Test mixing_frequency by accelerating PolyakSGD."""
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

    opt = PolyakSGD(fun=fun, max_stepsize=0.01, tol=0.05, momentum=False)
    history_size = 5
    aw = AndersonWrapper(opt, history_size=history_size, mixing_frequency=1)
    aw_params, aw_state = aw.run(pytree_init, l2reg=l2reg, data=data)
    self.assertLess(aw_state.error, 0.05)

  def test_optax_restart(self):
    """Test Optax optimizer."""
    X, y = datasets.make_classification(n_samples=100, n_features=20, n_classes=3,
                                        n_informative=3, random_state=0)
    data = (X, y)
    l2reg = 100.0
    # fun(params, data)
    fun = objective.l2_multiclass_logreg_with_intercept
    n_classes = len(jnp.unique(y))

    W_init = jnp.zeros((X.shape[1], n_classes))
    b_init = jnp.zeros(n_classes)
    pytree_init = (W_init, b_init)

    tol = 1e-2
    opt = OptaxSolver(opt=optax.sgd(1e-2, momentum=0.8), fun=fun, maxiter=1000, tol=0)
    aw = AndersonWrapper(opt, history_size=3, ridge=1e-3)
    params, infos = aw.run(pytree_init, l2reg=l2reg, data=data)

    # Check optimality conditions.
    error = opt.l2_optimality_error(params, l2reg=l2reg, data=data)
    self.assertLessEqual(error, tol)

  def test_block_cd_restart(self):
    """Accelerate Block CD."""
    X, y = datasets.make_regression(n_samples=10, n_features=3, random_state=0)

    # Setup parameters.
    fun = objective.least_squares  # fun(params, data)
    l1reg = 10.0
    data = (X, y)

    # Initialize.
    w_init = jnp.zeros(X.shape[1])
    tol = 5e-4
    maxiter = 100
    bcd = BlockCoordinateDescent(fun=fun, block_prox=prox.prox_lasso, tol=tol, maxiter=maxiter)
    aw = AndersonWrapper(bcd, history_size=3, ridge=1e-5)
    params, state = aw.run(init_params=w_init, hyperparams_prox=l1reg, data=data)

    # Check optimality conditions.
    self.assertLess(state.error, tol)

  def test_wrapper_grad(self):
    """Test gradient of wrapper."""
    data_train = datasets.make_regression(n_samples=100, n_features=3, random_state=0)
    fun = objective.least_squares
    lam = 10.0
    w_init = jnp.zeros(data_train[0].shape[1])
    tol = 1e-5
    maxiter = 1000  # large number of updates
    pg = ProximalGradient(fun=fun, prox=prox.prox_lasso, maxiter=maxiter, tol=tol,
                          acceleration=False)
    aw = AndersonWrapper(pg, history_size=5)
    data_val = datasets.make_regression(n_samples=100, n_features=3, random_state=0)

    def solve_run(lam):
      aw_params = aw.run(w_init, lam, data_train).params
      loss = fun(aw_params, data=data_val)
      return loss

    check_grads(solve_run, args=(lam,), order=1, modes=['rev'], eps=2e-2)

    def solve_run(lam):
      aw_params = aw.run(w_init, hyperparams_prox=lam, data=data_train).params
      loss = fun(aw_params, data=data_val)
      return loss

    check_grads(solve_run, args=(lam,), order=1, modes=['rev'], eps=2e-2)

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
