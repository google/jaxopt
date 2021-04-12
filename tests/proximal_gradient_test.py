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

from jaxopt import proximal_gradient
from jaxopt import implicit_diff
from jaxopt import projection
from jaxopt import test_util
from jaxopt import tree_util as tu
from jaxopt.prox import prox_elastic_net
from jaxopt.prox import prox_lasso

from sklearn import datasets
from sklearn import preprocessing


class ProximalGradientTest(jtu.JaxTestCase):

  @parameterized.product(acceleration=[True, False])
  def test_lasso(self, acceleration):
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.0
    tol = 1e-3 if acceleration else 5e-3
    maxiter = 200
    atol = 1e-2 if acceleration else 1e-1
    fun = test_util.make_least_squares_objective(X, y)

    # Check optimality conditions.
    w_init = jnp.zeros(X.shape[1])
    w_fit = proximal_gradient.proximal_gradient(
        fun,
        w_init,
        params_fun=lam,
        prox=prox_lasso,
        params_prox=1.0,
        tol=tol,
        maxiter=maxiter,
        acceleration=acceleration)
    w_fit2 = prox_lasso(w_fit - jax.grad(fun)(w_fit, lam), 1.0)
    self.assertLessEqual(jnp.sqrt(jnp.sum((w_fit - w_fit2)**2)), tol)

    # Compare against sklearn.
    w_skl = test_util.lasso_skl(X, y, lam)
    self.assertArraysAllClose(w_fit, w_skl, atol=atol)

  def test_lasso_with_intercept(self):
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.0
    tol = 1e-3
    maxiter = 200
    atol = 1e-2
    fun = test_util.make_least_squares_objective(X, y, fit_intercept=True)

    def prox(pytree, params, scaling=1.0):
      w, b = pytree
      return prox_lasso(w, params, scaling), b

    # Check optimality conditions.
    pytree_init = (jnp.zeros(X.shape[1]), 0.0)
    pytree_fit = proximal_gradient.proximal_gradient(
        fun,
        pytree_init,
        params_fun=lam,
        prox=prox,
        params_prox=1.0,
        tol=tol,
        maxiter=maxiter,
        acceleration=True)
    pytree = tu.tree_sub(pytree_fit, jax.grad(fun)(pytree_fit, lam))
    pytree_fit2 = prox(pytree, lam)
    pytree_fit_diff = tu.tree_sub(pytree_fit, pytree_fit2)
    self.assertLessEqual(tu.tree_l2_norm(pytree_fit_diff), tol)

    # Compare against sklearn.
    w_skl, b_skl = test_util.lasso_skl(X, y, lam, fit_intercept=True)
    self.assertArraysAllClose(pytree_fit[0], w_skl, atol=atol)
    self.assertAllClose(pytree_fit[1], b_skl, atol=atol)

  def test_lasso_implicit_diff(self):
    """Test implicit differentiation of a single lambda parameter."""
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.5
    tol = 1e-3
    maxiter = 200

    fun = test_util.make_least_squares_objective(X, y)
    jac_num = test_util.lasso_skl_jac(X, y, lam)
    w_skl = test_util.lasso_skl(X, y, lam)

    jac_fun = jax.jacrev(proximal_gradient.proximal_gradient, argnums=2)
    jac_custom = jac_fun(
        fun,
        w_skl,
        lam,
        prox_lasso,
        1.0,
        tol=tol,
        maxiter=maxiter,
        acceleration=True,
        implicit_diff=True)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-3)

    jac_fun = jax.jacrev(proximal_gradient.proximal_gradient, argnums=4)
    jac_custom = jac_fun(
        fun,
        w_skl,
        1.0,
        prox_lasso,
        lam,
        tol=tol,
        maxiter=maxiter,
        acceleration=True,
        implicit_diff=True)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-3)

  def test_lasso_implicit_diff_multi(self):
    """Test implicit differentiation of multiple lambda parameters."""
    X, y = datasets.make_regression(n_samples=50, n_features=3, random_state=0)

    # Use one lambda per feature.
    n_features = X.shape[1]
    lam = jnp.array([1.0, 5.0, 10.0])
    fun = test_util.make_least_squares_objective(X, y)

    # Compute solution.
    w_init = jnp.zeros_like(lam)
    tol = 1e-3
    maxiter = 200
    I = jnp.eye(n_features)
    proximal_gradient_fun = lambda lam_: proximal_gradient.proximal_gradient(
        fun=fun,
        init=w_init,
        params_fun=1.0,
        prox=prox_lasso,
        params_prox=lam_,
        tol=tol,
        maxiter=maxiter,
        acceleration=True,
        implicit_diff=True)
    sol = proximal_gradient_fun(lam)

    # Compute the Jacobian w.r.t. lam (params_prox) using VJPs.
    vjp_fun = lambda g: implicit_diff.pg_fixed_point_vjp(
        fun=fun,
        sol=sol,
        params_fun=1.0,
        prox=prox_lasso,
        params_prox=lam,
        cotangent=g)[1]
    jac_params_prox_from_vjp = jax.vmap(vjp_fun)(I)
    self.assertArraysEqual(jac_params_prox_from_vjp.shape,
                           (n_features, n_features))

    # Compute the Jacobian w.r.t. lam (params_prox) using JVPs.
    jvp_fun = lambda g: implicit_diff.pg_fixed_point_jvp(
        fun=fun,
        sol=sol,
        params_fun=1.0,
        prox=prox_lasso,
        params_prox=lam,
        tangents=(1.0, g))[1]
    jac_params_prox_from_jvp = jax.vmap(jvp_fun)(I)
    self.assertArraysAllClose(
        jac_params_prox_from_jvp, jac_params_prox_from_vjp, atol=tol)

    # Make sure the decorator works.
    jac_fun = jax.jacrev(proximal_gradient.proximal_gradient, argnums=4)
    jac_custom = jac_fun(
        fun,
        w_init,
        1.0,
        prox_lasso,
        lam,
        tol=tol,
        maxiter=maxiter,
        acceleration=True,
        implicit_diff=True)
    self.assertArraysAllClose(jac_params_prox_from_vjp, jac_custom, atol=tol)

  @parameterized.product(acceleration=[True, False])
  def test_lasso_forward_diff(self, acceleration):
    raise absltest.SkipTest
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.0
    tol = 1e-4 if acceleration else 5e-3
    maxiter = 200
    fun = test_util.make_least_squares_objective(X, y)

    jac_lam = test_util.lasso_skl_jac(X, y, lam)

    # Compute the Jacobian w.r.t. lam via forward differentiation.
    w_init = jnp.zeros(X.shape[1])
    jac_fun = jax.jacfwd(proximal_gradient.proximal_gradient, argnums=2)
    jac_lam2 = jac_fun(
        fun,
        w_init,
        lam,
        prox=prox_lasso,
        tol=tol,
        maxiter=maxiter,
        implicit_diff=False,
        acceleration=acceleration)
    self.assertArraysAllClose(jac_lam, jac_lam2, atol=1e-3)

  def test_elastic_net(self):
    X, y = datasets.load_boston(return_X_y=True)
    params_prox = (2.0, 0.8)
    tol = 1e-3
    maxiter = 200
    atol = 1e-3
    fun = test_util.make_least_squares_objective(X, y)
    prox = prox_elastic_net

    # Check optimality conditions.
    w_init = jnp.zeros(X.shape[1])
    w_fit = proximal_gradient.proximal_gradient(
        fun,
        w_init,
        params_fun=1.0,
        prox=prox,
        params_prox=params_prox,
        tol=tol,
        maxiter=maxiter)
    w_fit2 = prox(w_fit - jax.grad(fun)(w_fit, 1.0), params_prox)
    w_diff = tu.tree_sub(w_fit, w_fit2)
    self.assertLessEqual(jnp.sqrt(jnp.sum(w_diff**2)), tol)

    # Compare against sklearn.
    w_skl = test_util.enet_skl(X, y, *params_prox)
    self.assertArraysAllClose(w_fit, w_skl, atol=atol)

  def test_elastic_net_implicit_diff(self):
    """Test implicit differentiation of a single lambda parameter."""
    X, y = datasets.load_boston(return_X_y=True)
    params_prox = (2.0, 0.8)
    tol = 1e-3
    maxiter = 200
    fun = test_util.make_least_squares_objective(X, y)
    prox = prox_elastic_net

    jac_num_lam, jac_num_gam = test_util.enet_skl_jac(X, y, *params_prox)
    w_skl = test_util.enet_skl(X, y, *params_prox)
    jac_fun = jax.jacrev(proximal_gradient.proximal_gradient, argnums=4)
    jac_custom = jac_fun(
        fun,
        w_skl,
        1.0,
        prox,
        params_prox,
        tol=tol,
        maxiter=maxiter,
        acceleration=True,
        implicit_diff=True)
    self.assertArraysAllClose(jac_num_lam, jac_custom[0], atol=1e-3)
    self.assertArraysAllClose(jac_num_gam, jac_custom[1], atol=1e-3)

  def test_logreg(self):
    X, y = datasets.load_digits(return_X_y=True)
    lam = 1e2
    tol = 1e-4
    maxiter = 500
    atol = 1e-3
    fun = test_util.make_logreg_objective(X, y)

    W_init = jnp.zeros((X.shape[1], 10))
    W_fit = proximal_gradient.proximal_gradient(
        fun, W_init, params_fun=lam, tol=tol, maxiter=maxiter)

    # Check optimality conditions.
    W_grad = jax.grad(fun)(W_fit, lam)
    self.assertLessEqual(jnp.sqrt(jnp.sum(W_grad**2)), tol)

    # Compare against sklearn.
    W_skl = test_util.logreg_skl(X, y, lam)
    self.assertArraysAllClose(W_fit, W_skl, atol=atol)

  def test_logreg_with_intercept(self):
    X, y = datasets.load_digits(return_X_y=True)
    lam = 1e2
    tol = 1e-4
    maxiter = 200
    atol = 1e-3
    fun = test_util.make_logreg_objective(X, y, fit_intercept=True)

    pytree_init = (jnp.zeros((X.shape[1], 10)), jnp.zeros(10))
    pytree_fit = proximal_gradient.proximal_gradient(
        fun, pytree_init, params_fun=lam, prox=None, tol=tol, maxiter=maxiter)

    # Check optimality conditions.
    pytree_grad = jax.grad(fun)(pytree_fit, lam)
    self.assertLessEqual(tu.tree_l2_norm(pytree_grad), tol)

    # Compare against sklearn.
    W_skl, b_skl = test_util.logreg_skl(X, y, lam, fit_intercept=True)
    self.assertArraysAllClose(pytree_fit[0], W_skl, atol=atol)
    self.assertArraysAllClose(pytree_fit[1], b_skl, atol=atol)

  def test_logreg_implicit_diff(self):
    X, y = datasets.load_digits(return_X_y=True)
    lam = float(X.shape[0])
    tol = 1e-3
    maxiter = 200
    fun = test_util.make_logreg_objective(X, y)

    jac_num = test_util.logreg_skl_jac(X, y, lam)
    W_skl = test_util.logreg_skl(X, y, lam)

    # Make sure the decorator works.
    jac_fun = jax.jacrev(proximal_gradient.proximal_gradient, argnums=2)
    jac_custom = jac_fun(
        fun,
        W_skl,
        lam,
        prox=None,
        tol=tol,
        maxiter=maxiter,
        acceleration=True,
        implicit_diff=True)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-2)

  @parameterized.product(acceleration=[True, False])
  def test_logreg_forward_diff(self, acceleration):
    X, y = datasets.load_digits(return_X_y=True)
    lam = float(X.shape[0])
    tol = 1e-3 if acceleration else 5e-3
    maxiter = 200
    atol = 1e-3 if acceleration else 1e-1
    fun = test_util.make_logreg_objective(X, y)

    jac_lam = test_util.logreg_skl_jac(X, y, lam)

    # Compute the Jacobian w.r.t. lam via forward differentiation.
    W_init = jnp.zeros((X.shape[1], 10))
    jac_fun = jax.jacfwd(proximal_gradient.proximal_gradient, argnums=2)
    jac_lam2 = jac_fun(
        fun,
        W_init,
        lam,
        prox=None,
        tol=tol,
        maxiter=maxiter,
        implicit_diff=False,
        acceleration=acceleration)
    self.assertArraysAllClose(jac_lam, jac_lam2, atol=atol)

  def test_multiclass_svm_dual(self):
    X, y = datasets.make_classification(
        n_samples=20,
        n_features=5,
        n_informative=3,
        n_classes=3,
        random_state=0)
    # Transform labels to a one-hot representation.
    # Y has shape (n_samples, n_classes).
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    lam = 10.0
    tol = 1e-2
    maxiter = 500
    atol = 1e-2
    fun = test_util.make_multiclass_linear_svm_objective(X, y)

    proj_vmap = jax.vmap(projection.projection_simplex)
    prox = lambda x, params_prox, scaling=1.0: proj_vmap(x)

    n_samples, n_classes = Y.shape
    beta_init = jnp.ones((n_samples, n_classes)) / n_classes
    beta_fit = proximal_gradient.proximal_gradient(
        fun,
        beta_init,
        params_fun=lam,
        prox=prox,
        stepsize=1e-2,
        tol=tol,
        maxiter=maxiter)

    # Check optimality conditions.
    beta_fit2 = proj_vmap(beta_fit - jax.grad(fun)(beta_fit, lam))
    self.assertLessEqual(jnp.sqrt(jnp.sum((beta_fit - beta_fit2)**2)), tol)

    # Compare against sklearn.
    W_skl = test_util.multiclass_linear_svm_skl(X, y, lam)
    W_fit = jnp.dot(X.T, (Y - beta_fit)) / lam
    self.assertArraysAllClose(W_fit, W_skl, atol=atol)

  def test_multiclass_svm_dual_implicit_diff(self):

    raise absltest.SkipTest
    X, y = datasets.make_classification(
        n_samples=20,
        n_features=5,
        n_informative=3,
        n_classes=3,
        random_state=0)
    # Transform labels to a one-hot representation.
    # Y has shape (n_samples, n_classes).
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    lam = 10.0
    tol = 1e-3
    maxiter = 500
    atol = 1e-2
    fun = test_util.make_multiclass_linear_svm_objective(X, y)

    proj_vmap = jax.vmap(projection.projection_simplex)
    prox = lambda x, params_prox, scaling=1.0: proj_vmap(x)

    def proximal_gradient_fun_dual(lam):
      n_samples, n_classes = Y.shape
      beta_init = jnp.ones((n_samples, n_classes)) / n_classes
      return proximal_gradient.proximal_gradient(
          fun,
          beta_init,
          params_fun=lam,
          prox=prox,
          stepsize=1e-2,
          tol=tol,
          maxiter=maxiter)

    def proximal_gradient_fun_primal(lam):
      beta_fit = proximal_gradient_fun_dual(lam)
      return jnp.dot(X.T, (Y - beta_fit)) / lam

    jac_primal = jax.jacrev(proximal_gradient_fun_primal)(lam)
    jac_num = test_util.multiclass_linear_svm_skl_jac(X, y, lam)
    self.assertArraysAllClose(jac_num, jac_primal, atol=1e-3)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  # jax.config.update("jax_enable_x64", True)
  absltest.main(testLoader=jtu.JaxTestLoader())
