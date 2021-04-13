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


from jaxopt.implicit_diff import gd_fixed_point
from jaxopt.implicit_diff import gd_fixed_point_jvp
from jaxopt.implicit_diff import gd_fixed_point_vjp

from jaxopt.implicit_diff import pg_fixed_point
from jaxopt.implicit_diff import pg_fixed_point_jvp

from jaxopt.implicit_diff import pg_fixed_point_vjp
from jaxopt.prox import prox_elastic_net
from jaxopt.prox import prox_lasso
from jaxopt import test_util

from sklearn import datasets


class ImplicitDiffTest(jtu.JaxTestCase):

  def test_lasso(self):
    X, y = datasets.make_regression(n_samples=50, n_features=10, random_state=0)
    lam = 10.0
    w_skl = test_util.lasso_skl(X, y, lam, fit_intercept=False)
    jac_num = test_util.lasso_skl_jac(X, y, lam, fit_intercept=False)
    fun = test_util.make_least_squares_objective(X, y, fit_intercept=False)

    # Compute the Jacobian w.r.t. lam (params_fun) using VJPs.
    I = jnp.eye(len(w_skl))
    vjp_fun = lambda g: pg_fixed_point_vjp(fun=fun, sol=w_skl, params_fun=lam,
                                           prox=prox_lasso, params_prox=1.0,
                                           cotangent=g)[0]
    jac_params_fun = jax.vmap(vjp_fun)(I)
    self.assertArraysAllClose(jac_num, jac_params_fun, atol=1e-4)

    # Same but now w.r.t. params_prox.
    vjp_fun = lambda g: pg_fixed_point_vjp(fun=fun, sol=w_skl, params_fun=1.0,
                                           prox=prox_lasso, params_prox=lam,
                                           cotangent=g)[1]
    jac_params_prox = jax.vmap(vjp_fun)(I)
    self.assertArraysAllClose(jac_num, jac_params_prox, atol=1e-4)

    # Compute the Jacobian w.r.t. lam (params_fun) using JVPs.
    jac_params_fun = pg_fixed_point_jvp(fun=fun, sol=w_skl, params_fun=lam,
                                        prox=prox_lasso, params_prox=1.0,
                                        tangents=(1.0, 1.0))[0]
    self.assertArraysAllClose(jac_num, jac_params_fun, atol=1e-4)

    # Same but now w.r.t. params_prox.
    jac_params_prox = pg_fixed_point_jvp(fun=fun, sol=w_skl, params_fun=1.0,
                                         prox=prox_lasso, params_prox=lam,
                                         tangents=(1.0, 1.0))[1]
    self.assertArraysAllClose(jac_num, jac_params_prox, atol=1e-4)

  def test_lasso_wrapper(self):
    X, y = datasets.make_regression(n_samples=50, n_features=10, random_state=0)
    lam = 10.0
    fun = test_util.make_least_squares_objective(X, y, fit_intercept=False)

    @pg_fixed_point(fun, prox_lasso)
    def solver_fun(params_fun, params_prox):
      return test_util.lasso_skl(X, y, params_prox, fit_intercept=False)

    jac_num = test_util.lasso_skl_jac(X, y, lam, fit_intercept=False)
    jac_lam = jax.jacrev(solver_fun, argnums=1)(1.0, lam)
    self.assertArraysAllClose(jac_num, jac_lam, atol=1e-4)

  def test_elastic_net(self):
    X, y = datasets.make_regression(n_samples=50, n_features=10, random_state=0)
    params_prox = (2.0, 0.8)
    fun = test_util.make_least_squares_objective(X, y, fit_intercept=False)
    prox = prox_elastic_net

    # Jacobian w.r.t. lam using central finite difference.
    # We use the sklearn solver for precision, as it operates on float64.
    jac_num_lam, jac_num_gam = test_util.enet_skl_jac(X, y,
                                                      params_prox[0],
                                                      params_prox[1])
    w_skl = test_util.enet_skl(X, y, params_prox[0], params_prox[1])

    # Compute the Jacobian w.r.t. params_prox using VJPs.
    vjp_fun = lambda g: pg_fixed_point_vjp(fun=fun, sol=w_skl, params_fun=1.0,
                                           prox=prox, params_prox=params_prox,
                                           cotangent=g)[1]
    I = jnp.eye(len(w_skl))
    jac_params_prox = jax.vmap(vjp_fun)(I)
    self.assertArraysAllClose(jac_num_lam, jac_params_prox[0], atol=1e-4)
    self.assertArraysAllClose(jac_num_gam, jac_params_prox[1], atol=1e-4)

    # Compute the Jacobian w.r.t. params_prox using JVPs.
    jvp_fun = lambda g: pg_fixed_point_jvp(fun=fun, sol=w_skl, params_fun=1.0,
                                           prox=prox, params_prox=params_prox,
                                           tangents=(1.0, g))[1]
    jac_lam = jvp_fun((1.0, 0.0))
    jac_gam = jvp_fun((0.0, 1.0))
    self.assertArraysAllClose(jac_num_lam, jac_lam, atol=1e-4)
    self.assertArraysAllClose(jac_num_gam, jac_gam, atol=1e-4)

  def test_logreg(self):
    X, y = datasets.make_classification(n_samples=50, n_features=10,
                                        n_informative=5, n_classes=3,
                                        random_state=0)
    lam = 1.0

    W_skl = test_util.logreg_skl(X, y, lam, fit_intercept=False)
    jac_num = test_util.logreg_skl_jac(X, y, lam, fit_intercept=False)
    fun = test_util.make_logreg_objective(X, y, fit_intercept=False)

    # Compute the Jacobian w.r.t. lam (params_fun) via implicit VJPs.
    I = jnp.eye(W_skl.size)
    I = I.reshape(-1, *W_skl.shape)
    vjp_fun = lambda g: gd_fixed_point_vjp(fun=fun, sol=W_skl, cotangent=g,
                                           params_fun=lam)
    jac_lam = jax.vmap(vjp_fun)(I).reshape(*W_skl.shape)
    self.assertArraysAllClose(jac_num, jac_lam, atol=1e-4)

    # Compute the Jacobian w.r.t. lam (params_fun) using JVPs.
    jac_lam2 = gd_fixed_point_jvp(fun=fun, sol=W_skl, params_fun=lam,
                                  tangent=1.0)
    self.assertArraysAllClose(jac_num, jac_lam2, atol=1e-4)

  def test_logreg_wrapper(self):
    X, y = datasets.make_classification(n_samples=50, n_features=10,
                                        n_informative=5, n_classes=3,
                                        random_state=0)
    lam = 1.0
    fun = test_util.make_logreg_objective(X, y, fit_intercept=False)

    @gd_fixed_point(fun)
    def solver_fun(lam):
      return test_util.logreg_skl(X, y, lam, fit_intercept=False)

    jac_num = test_util.logreg_skl_jac(X, y, lam, fit_intercept=False)
    jac_lam = jax.jacrev(solver_fun)(lam)
    self.assertArraysAllClose(jac_num, jac_lam, atol=1e-4)

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
