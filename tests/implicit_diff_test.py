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

from jaxopt.implicit_diff import fixed_point_jvp
from jaxopt.implicit_diff import fixed_point_vjp
from jaxopt.implicit_diff import custom_fixed_point
from jaxopt.implicit_diff import make_block_cd_fixed_point_fun
from jaxopt.implicit_diff import make_gradient_descent_fixed_point_fun
from jaxopt.implicit_diff import make_mirror_descent_fixed_point_fun
from jaxopt.implicit_diff import make_proximal_gradient_fixed_point_fun

from jaxopt.projection import projection_simplex
from jaxopt import proximal_gradient
from jaxopt.prox import prox_elastic_net
from jaxopt.prox import prox_lasso
from jaxopt import test_util

from sklearn import datasets
from sklearn import preprocessing


class ImplicitDiffTest(jtu.JaxTestCase):

  @parameterized.product(fixed_point_fun=["pg", "bcd"])
  def test_lasso(self, fixed_point_fun):
    X, y = datasets.make_regression(n_samples=50, n_features=10, random_state=0)
    lam = 10.0
    w_skl = test_util.lasso_skl(X, y, lam, fit_intercept=False)
    jac_num = test_util.lasso_skl_jac(X, y, lam, fit_intercept=False)
    fun = test_util.make_least_squares_objective(X, y, fit_intercept=False)
    if fixed_point_fun == "pg":
      fixed_point_fun = make_proximal_gradient_fixed_point_fun(fun, prox_lasso)
    else:
      fixed_point_fun = make_block_cd_fixed_point_fun(fun, prox_lasso)

    # Compute the Jacobian w.r.t. lam (params_fun) using VJPs.
    I = jnp.eye(len(w_skl))
    vjp_fun = lambda g: fixed_point_vjp(fixed_point_fun=fixed_point_fun,
                                        sol=w_skl, params=(lam, 1.0),
                                        cotangent=g)[0]
    jac_params_fun = jax.vmap(vjp_fun)(I)
    self.assertArraysAllClose(jac_num, jac_params_fun, atol=1e-4)

    # Same but now w.r.t. params_prox.
    vjp_fun = lambda g: fixed_point_vjp(fixed_point_fun=fixed_point_fun,
                                        sol=w_skl, params=(1.0, lam),
                                        cotangent=g)[1]
    jac_params_prox = jax.vmap(vjp_fun)(I)
    self.assertArraysAllClose(jac_num, jac_params_prox, atol=1e-4)

    # Compute the Jacobian w.r.t. lam (params_fun) using JVPs.
    fixed_point_fun1 = lambda sol, pf: fixed_point_fun(sol, (pf, 1.0))
    jac_params_fun = fixed_point_jvp(fixed_point_fun=fixed_point_fun1,
                                     sol=w_skl, params=lam,
                                     tangent=1.0)
    self.assertArraysAllClose(jac_num, jac_params_fun, atol=1e-4)

    # Same but now w.r.t. params_prox.
    fixed_point_fun2 = lambda sol, pp: fixed_point_fun(sol, (1.0, pp))
    jac_params_prox = fixed_point_jvp(fixed_point_fun=fixed_point_fun2,
                                      sol=w_skl, params=lam,
                                      tangent=1.0)
    self.assertArraysAllClose(jac_num, jac_params_prox, atol=1e-4)

  def test_lasso_wrapper(self):
    X, y = datasets.make_regression(n_samples=50, n_features=10, random_state=0)
    lam = 10.0
    fun = test_util.make_least_squares_objective(X, y, fit_intercept=False)
    fixed_point_fun = make_proximal_gradient_fixed_point_fun(fun, prox_lasso)

    @custom_fixed_point(fixed_point_fun, unpack_params=True)
    def solver_fun(params_fun, params_prox):
      return test_util.lasso_skl(X, y, params_prox, fit_intercept=False)

    jac_num = test_util.lasso_skl_jac(X, y, lam, fit_intercept=False)
    jac_lam = jax.jacrev(solver_fun, argnums=1)(1.0, lam)
    self.assertArraysAllClose(jac_num, jac_lam, atol=1e-4)

    @custom_fixed_point(fixed_point_fun, unpack_params=False)
    def solver_fun2(params):
      _, params_prox = params
      return test_util.lasso_skl(X, y, params_prox, fit_intercept=False)

    _, jac_lam2 = jax.jacrev(solver_fun2)((1.0, lam))
    self.assertArraysAllClose(jac_num, jac_lam2, atol=1e-4)

  def test_elastic_net(self):
    X, y = datasets.make_regression(n_samples=50, n_features=10, random_state=0)
    params_prox = (2.0, 0.8)
    fun = test_util.make_least_squares_objective(X, y, fit_intercept=False)
    prox = prox_elastic_net
    fixed_point_fun = make_proximal_gradient_fixed_point_fun(fun, prox)

    # Jacobian w.r.t. lam using central finite difference.
    # We use the sklearn solver for precision, as it operates on float64.
    jac_num_lam, jac_num_gam = test_util.enet_skl_jac(X, y, params_prox)
    w_skl = test_util.enet_skl(X, y, params_prox)

    # Compute the Jacobian w.r.t. params_prox using VJPs.
    vjp_fun = lambda g: fixed_point_vjp(fixed_point_fun=fixed_point_fun,
                                        sol=w_skl,
                                        params=(1.0, params_prox),
                                        cotangent=g)[1]
    I = jnp.eye(len(w_skl))
    jac_params_prox = jax.vmap(vjp_fun)(I)
    self.assertArraysAllClose(jac_num_lam, jac_params_prox[0], atol=1e-4)
    self.assertArraysAllClose(jac_num_gam, jac_params_prox[1], atol=1e-4)

    # Compute the Jacobian w.r.t. params_prox using JVPs.
    fixed_point_fun2 = lambda sol, pp: fixed_point_fun(sol, (1.0, pp))
    jvp_fun = lambda g: fixed_point_jvp(fixed_point_fun=fixed_point_fun2,
                                        sol=w_skl,
                                        params=params_prox,
                                        tangent=g)
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
    fixed_point_fun = make_gradient_descent_fixed_point_fun(fun)

    # Compute the Jacobian w.r.t. lam (params_fun) via implicit VJPs.
    I = jnp.eye(W_skl.size)
    I = I.reshape(-1, *W_skl.shape)
    vjp_fun = lambda g: fixed_point_vjp(fixed_point_fun=fixed_point_fun,
                                        sol=W_skl, params=lam, cotangent=g)
    jac_lam = jax.vmap(vjp_fun)(I).reshape(*W_skl.shape)
    self.assertArraysAllClose(jac_num, jac_lam, atol=1e-4)

    # Compute the Jacobian w.r.t. lam (params_fun) using JVPs.
    jac_lam2 = fixed_point_jvp(fixed_point_fun=fixed_point_fun,
                               sol=W_skl, params=lam, tangent=1.0)
    self.assertArraysAllClose(jac_num, jac_lam2, atol=1e-4)

  def test_logreg_wrapper(self):
    X, y = datasets.make_classification(n_samples=50, n_features=10,
                                        n_informative=5, n_classes=3,
                                        random_state=0)
    lam = 1.0
    fun = test_util.make_logreg_objective(X, y, fit_intercept=False)
    fixed_point_fun = make_gradient_descent_fixed_point_fun(fun)

    @custom_fixed_point(fixed_point_fun)
    def solver_fun(lam):
      return test_util.logreg_skl(X, y, lam, fit_intercept=False)

    jac_num = test_util.logreg_skl_jac(X, y, lam, fit_intercept=False)
    jac_lam = jax.jacrev(solver_fun)(lam)
    self.assertArraysAllClose(jac_num, jac_lam, atol=1e-4)

  def test_multiclass_svm_dual(self):
    X, y = datasets.make_classification(n_samples=20, n_features=5,
                                        n_informative=3, n_classes=3,
                                        random_state=0)
    X = preprocessing.Normalizer().fit_transform(X)
    # Transform labels to a one-hot representation.
    # Y has shape (n_samples, n_classes).
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    lam = 10.0
    tol = 1e-3
    maxiter = 500
    fun = test_util.make_multiclass_linear_svm_objective(X, y)
    proj_vmap = jax.vmap(projection_simplex)
    prox = lambda x, params_prox, scaling=1.0: proj_vmap(x)
    fixed_point_fun = make_proximal_gradient_fixed_point_fun(fun, prox)

    # Commpute solution using proximal_gradient.
    n_samples, n_classes = Y.shape
    beta_init = jnp.ones((n_samples, n_classes)) / n_classes
    solver_fun = proximal_gradient.make_solver_fun(fun=fun, init=beta_init,
                                                   prox=prox, stepsize=1e-2,
                                                   tol=tol, maxiter=maxiter)
    beta_fit = solver_fun(params_fun=lam)

    # Compute the Jacobian w.r.t. lam (params_fun) using the PG fixed point.
    I = jnp.eye(beta_fit.size)
    I = I.reshape(-1, *beta_fit.shape)

    pg_fp_fun = make_proximal_gradient_fixed_point_fun(fun, prox)
    vjp_fun_pg = lambda g: fixed_point_vjp(fixed_point_fun=pg_fp_fun,
                                           sol=beta_fit, params=(lam, 1.0),
                                           cotangent=g)[0]
    jac_lam_pg = jax.vmap(vjp_fun_pg)(I).reshape(*beta_fit.shape)

    # Compute the Jacobian w.r.t. lam (params_fun) using the MD fixed point.

    # Row-wise KL projections on the simplex.
    softmax_vmap = jax.vmap(jax.nn.softmax)
    # The proj function needs to support a parameter (unused here).
    proj = lambda x, _: softmax_vmap(x)
    # Generating function of the Bregman divergence.
    generating_fun = lambda x: -jnp.sum(jax.scipy.special.entr(x))
    # Row-wise mapping.
    mapping_fun = jax.vmap(jax.grad(generating_fun))

    md_fp_fun = make_mirror_descent_fixed_point_fun(fun, proj, mapping_fun)
    vjp_fun_md = lambda g: fixed_point_vjp(fixed_point_fun=md_fp_fun,
                                           sol=beta_fit, params=(lam, 1.0),
                                           cotangent=g)[0]
    jac_lam_md = jax.vmap(vjp_fun_md)(I).reshape(*beta_fit.shape)
    self.assertArraysAllClose(jac_lam_pg, jac_lam_md, atol=5e-2)

if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
