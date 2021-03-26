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

from jaxopt import fista
from jaxopt.implicit_diff import prox_fixed_point_jvp
from jaxopt.implicit_diff import prox_fixed_point_vjp
from jaxopt import loss
from jaxopt import projection
from jaxopt import prox
from jaxopt import tree_util as tu

from sklearn import datasets
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm


def _lasso_skl(X, y, lam, tol=1e-5, fit_intercept=False):
  X = preprocessing.Normalizer().fit_transform(X)
  lasso = linear_model.Lasso(fit_intercept=fit_intercept, alpha=lam, tol=tol)
  lasso = lasso.fit(X, y)
  if fit_intercept:
    return lasso.coef_, lasso.intercept_
  else:
    return lasso.coef_


def _enet_skl(X, y, lam, gamma, tol=1e-5, fit_intercept=False):
  alpha = lam + lam * gamma
  l1_ratio = lam / alpha
  X = preprocessing.Normalizer().fit_transform(X)
  enet = linear_model.ElasticNet(fit_intercept=fit_intercept, alpha=alpha,
                                  l1_ratio=l1_ratio, tol=tol)
  enet = enet.fit(X, y)
  if fit_intercept:
    return enet.coef_, enet.intercept_
  else:
    return enet.coef_


def _make_lasso_objective(X, y, fit_intercept=False):
  X = preprocessing.Normalizer().fit_transform(X)
  if fit_intercept:
    def fun_f(pytree, lam):
      w, b = pytree
      y_pred = jnp.dot(X, w) + b
      diff = y_pred - y
      return 0.5 / (lam * X.shape[0]) * jnp.dot(diff, diff)
  else:
    def fun_f(w, lam):
      y_pred = jnp.dot(X, w)
      diff = y_pred - y
      return 0.5 / (lam * X.shape[0]) * jnp.dot(diff, diff)
  return fun_f


def _logreg_skl(X, y, lam, tol=1e-5, fit_intercept=False):
  X = preprocessing.Normalizer().fit_transform(X)
  logreg = linear_model.LogisticRegression(fit_intercept=fit_intercept,
                                           C=1. / lam, tol=tol,
                                           multi_class="multinomial")
  logreg = logreg.fit(X, y)
  if fit_intercept:
    return logreg.coef_.T, logreg.intercept_
  else:
    return logreg.coef_.T


def _make_logreg_objective(X, y, fit_intercept=False):
  X = preprocessing.Normalizer().fit_transform(X)
  if fit_intercept:
    def fun_f(pytree, lam):
      W, b = pytree
      logits = jnp.dot(X, W) + b
      return (jnp.sum(jax.vmap(loss.multiclass_logistic_loss)(y, logits)) +
              0.5 * lam * jnp.sum(W ** 2))
  else:
    def fun_f(W, lam):
      logits = jnp.dot(X, W)
      return (jnp.sum(jax.vmap(loss.multiclass_logistic_loss)(y, logits)) +
              0.5 * lam * jnp.sum(W ** 2))
  return fun_f


class FISTATest(jtu.JaxTestCase):

  @parameterized.product(acceleration=[True, False],
                         implicit_diff=[True, False])
  def test_lasso(self, acceleration, implicit_diff):
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.0
    tol = 1e-3 if acceleration else 5e-3
    max_iter = 200
    atol = 1e-2 if acceleration else 1e-1
    fun_f = _make_lasso_objective(X, y)

    # Check optimality conditions.
    w_init = jnp.zeros(X.shape[1])
    w_fit = fista.fista(fun_f, w_init, params_f=lam, prox_g=prox.prox_lasso,
                        params_g=1.0, tol=tol, max_iter=max_iter,
                        acceleration=acceleration, implicit_diff=implicit_diff)
    w_fit2 = prox.prox_lasso(w_fit - jax.grad(fun_f)(w_fit, lam), 1.0)
    self.assertLessEqual(jnp.sqrt(jnp.sum((w_fit - w_fit2) ** 2)), tol)

    # Compare against sklearn.
    w_skl = _lasso_skl(X, y, lam)
    self.assertArraysAllClose(w_fit, w_skl, atol=atol)

  def test_lasso_with_intercept(self):
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.0
    tol = 1e-3
    max_iter = 200
    atol = 1e-2
    fun_f = _make_lasso_objective(X, y, fit_intercept=True)

    def prox_g(pytree, params, scaling=1.0):
      w, b = pytree
      return prox.prox_lasso(w, params, scaling), b

    # Check optimality conditions.
    pytree_init = (jnp.zeros(X.shape[1]), 0.0)
    pytree_fit = fista.fista(fun_f, pytree_init, params_f=lam, prox_g=prox_g,
                             params_g=1.0, tol=tol, max_iter=max_iter,
                             acceleration=True)
    pytree = tu.tree_sub(pytree_fit, jax.grad(fun_f)(pytree_fit, lam))
    pytree_fit2 = prox_g(pytree, lam)
    pytree_fit_diff = tu.tree_sub(pytree_fit, pytree_fit2)
    self.assertLessEqual(tu.tree_l2_norm(pytree_fit_diff), tol)

    # Compare against sklearn.
    w_skl, b_skl = _lasso_skl(X, y, lam, fit_intercept=True)
    self.assertArraysAllClose(pytree_fit[0], w_skl, atol=atol)
    self.assertAllClose(pytree_fit[1], b_skl, atol=atol)

  def test_lasso_implicit_diff(self):
    """Test implicit differentiation of a single lambda parameter."""
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.5
    eps = 1e-5
    fun_f = _make_lasso_objective(X, y)

    # Jacobian w.r.t. lam using central finite difference.
    # We use the sklearn solver for precision, as it operates on float64.
    jac_num = (_lasso_skl(X, y, lam + eps) -
               _lasso_skl(X, y, lam - eps)) / (2 * eps)

    # Compute the Jacobian w.r.t. lam (params_f) using VJPs.
    w_skl = _lasso_skl(X, y, lam)
    I = jnp.eye(len(w_skl))
    fun = lambda g: prox_fixed_point_vjp(fun_f=fun_f, sol=w_skl, params_f=lam,
                                         prox_g=prox.prox_lasso, params_g=1.0,
                                         cotangent=g)[0]
    jac_params_f = jax.vmap(fun)(I)
    self.assertArraysAllClose(jac_num, jac_params_f, atol=1e-3)

    # Same but now w.r.t. params_g.
    fun = lambda g: prox_fixed_point_vjp(fun_f=fun_f, sol=w_skl, params_f=1.0,
                                         prox_g=prox.prox_lasso, params_g=lam,
                                         cotangent=g)[1]
    jac_params_g = jax.vmap(fun)(I)
    self.assertArraysAllClose(jac_num, jac_params_g, atol=1e-3)

    # Compute the Jacobian w.r.t. lam (params_f) using JVPs.
    jac_params_f = prox_fixed_point_jvp(fun_f=fun_f, sol=w_skl, params_f=lam,
                                        prox_g=prox.prox_lasso, params_g=1.0,
                                        tangents=(1.0, 1.0))[0]
    self.assertArraysAllClose(jac_num, jac_params_f, atol=1e-3)

    # Same but now w.r.t. params_g.
    jac_params_g = prox_fixed_point_jvp(fun_f=fun_f, sol=w_skl, params_f=1.0,
                                        prox_g=prox.prox_lasso, params_g=lam,
                                        tangents=(1.0, 1.0))[1]
    self.assertArraysAllClose(jac_num, jac_params_g, atol=1e-3)

    # Make sure the decorator works.
    w_init = jnp.zeros(X.shape[1])
    tol = 1e-3
    max_iter = 200

    jac_fun = jax.jacrev(fista.fista, argnums=2)
    jac_custom = jac_fun(fun_f, w_skl, lam, prox.prox_lasso, 1.0,
                         tol=tol, max_iter=max_iter, acceleration=True,
                         implicit_diff=True)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-3)


    jac_fun = jax.jacrev(fista.fista, argnums=4)
    jac_custom = jac_fun(fun_f, w_skl, 1.0, prox.prox_lasso, lam,
                         tol=tol, max_iter=max_iter, acceleration=True,
                         implicit_diff=True)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-3)

  def test_lasso_implicit_diff_multi(self):
    """Test implicit differentiation of multiple lambda parameters."""
    X, y = datasets.make_regression(n_samples=50, n_features=3, random_state=0)

    # Use one lambda per feature.
    n_features = X.shape[1]
    lam = jnp.array([1.0, 5.0, 10.0])
    eps = 1e-4
    fun_f = _make_lasso_objective(X, y)

    # Compute solution.
    w_init = jnp.zeros_like(lam)
    tol = 1e-3
    max_iter = 200
    I = jnp.eye(n_features)
    fista_fun = lambda lam_: fista.fista(fun_f=fun_f, init=w_init, params_f=1.0,
                                         prox_g=prox.prox_lasso, params_g=lam_,
                                         tol=tol, max_iter=max_iter,
                                         acceleration=True, implicit_diff=True)
    sol = fista_fun(lam)

    # Compute the Jacobian w.r.t. lam (params_g) using VJPs.
    fun = lambda g: prox_fixed_point_vjp(fun_f=fun_f, sol=sol, params_f=1.0,
                                         prox_g=prox.prox_lasso, params_g=lam,
                                         cotangent=g)[1]
    jac_params_g_from_vjp = jax.vmap(fun)(I)
    self.assertArraysEqual(jac_params_g_from_vjp.shape,
                           (n_features, n_features))

    # Compute the Jacobian w.r.t. lam (params_g) using JVPs.
    fun = lambda g: prox_fixed_point_jvp(fun_f=fun_f, sol=sol, params_f=1.0,
                                         prox_g=prox.prox_lasso, params_g=lam,
                                         tangents=(1.0, g))[1]
    jac_params_g_from_jvp = jax.vmap(fun)(I)
    self.assertArraysAllClose(jac_params_g_from_jvp, jac_params_g_from_vjp,
                              atol=tol)

    # Make sure the decorator works.
    jac_fun = jax.jacrev(fista.fista, argnums=4)
    jac_custom = jac_fun(fun_f, w_init, 1.0, prox.prox_lasso, lam,
                         tol=tol, max_iter=max_iter, acceleration=True,
                         implicit_diff=True)
    self.assertArraysAllClose(jac_params_g_from_vjp, jac_custom, atol=tol)

  @parameterized.product(acceleration=[True, False])
  def test_lasso_forward_diff(self, acceleration):
    raise absltest.SkipTest
    X, y = datasets.load_boston(return_X_y=True)
    lam = 1.0
    tol = 1e-4 if acceleration else 5e-3
    max_iter = 200
    eps = 1e-5
    fun_f = _make_lasso_objective(X, y)

    jac_lam = (_lasso_skl(X, y, lam + eps) -
               _lasso_skl(X, y, lam - eps)) / (2 * eps)

    # Compute the Jacobian w.r.t. lam via forward differentiation.
    w_init = jnp.zeros(X.shape[1])
    jac_fun = jax.jacfwd(fista.fista, argnums=2)
    jac_lam2 = jac_fun(fun_f, w_init, lam, prox_g=prox.prox_lasso, tol=tol,
                       max_iter=max_iter, implicit_diff=False,
                       acceleration=acceleration)
    self.assertArraysAllClose(jac_lam, jac_lam2, atol=1e-3)

  def test_elastic_net(self):
    X, y = datasets.load_boston(return_X_y=True)
    params_g = (2.0, 0.8)
    tol = 1e-3
    max_iter = 200
    atol = 1e-3
    fun_f = _make_lasso_objective(X, y)
    prox_g = prox.prox_elastic_net

    # Check optimality conditions.
    w_init = jnp.zeros(X.shape[1])
    w_fit = fista.fista(fun_f, w_init, params_f=1.0, prox_g=prox_g,
                        params_g=params_g, tol=tol,
                        implicit_diff=False, verbose=0,
                        max_iter=max_iter)
    w_fit2 = prox_g(w_fit - jax.grad(fun_f)(w_fit, 1.0), params_g)
    w_diff = tu.tree_sub(w_fit, w_fit2)
    self.assertLessEqual(jnp.sqrt(jnp.sum(w_diff ** 2)), tol)

    # Compare against sklearn.
    w_skl = _enet_skl(X, y, lam=params_g[0], gamma=params_g[1])
    self.assertArraysAllClose(w_fit, w_skl, atol=atol)

  def test_elastic_net_implicit_diff(self):
    """Test implicit differentiation of a single lambda parameter."""
    X, y = datasets.load_boston(return_X_y=True)
    params_g = (2.0, 0.8)
    eps = 1e-5
    fun_f = _make_lasso_objective(X, y)
    prox_g = prox.prox_elastic_net

    # Jacobian w.r.t. lam using central finite difference.
    # We use the sklearn solver for precision, as it operates on float64.
    jac_num_lam = (_enet_skl(X, y, params_g[0] + eps, params_g[1]) -
                   _enet_skl(X, y, params_g[0] - eps, params_g[1])) / (2 * eps)
    jac_num_gam = (_enet_skl(X, y, params_g[0], params_g[1] + eps) -
                   _enet_skl(X, y, params_g[0], params_g[1] - eps)) / (2 * eps)

    w_skl = _enet_skl(X, y, params_g[0], params_g[1])

    # Compute the Jacobian w.r.t. params_g using VJPS.
    fun = lambda g: prox_fixed_point_vjp(fun_f=fun_f, sol=w_skl, params_f=1.0,
                                         prox_g=prox_g, params_g=params_g,
                                         cotangent=g)[1]
    I = jnp.eye(len(w_skl))
    jac_params_g = jax.vmap(fun)(I)
    self.assertArraysAllClose(jac_num_lam, jac_params_g[0], atol=1e-3)
    self.assertArraysAllClose(jac_num_gam, jac_params_g[1], atol=1e-3)

    # Compute the Jacobian w.r.t. params_g using JVPs.
    fun = lambda g: prox_fixed_point_jvp(fun_f=fun_f, sol=w_skl, params_f=1.0,
                                         prox_g=prox_g, params_g=params_g,
                                         tangents=(1.0, g))[1]
    jac_lam = fun((1.0, 0.0))
    jac_gam = fun((0.0, 1.0))
    self.assertArraysAllClose(jac_num_lam, jac_lam, atol=1e-3)
    self.assertArraysAllClose(jac_num_gam, jac_gam, atol=1e-3)

    # Make sure the decorator works.
    w_init = jnp.zeros(X.shape[1])
    tol = 1e-3
    max_iter = 200

    jac_fun = jax.jacrev(fista.fista, argnums=4)
    jac_custom = jac_fun(fun_f, w_skl, 1.0, prox_g, params_g,
                         tol=tol, max_iter=max_iter, acceleration=True,
                         implicit_diff=True)
    self.assertArraysAllClose(jac_num_lam, jac_custom[0], atol=1e-3)
    self.assertArraysAllClose(jac_num_gam, jac_custom[1], atol=1e-3)

  def test_logreg(self):
    X, y = datasets.load_digits(return_X_y=True)
    lam = float(X.shape[0])
    tol = 1e-3
    max_iter = 200
    atol = 1e-3
    fun_f = _make_logreg_objective(X, y)

    W_init = jnp.zeros((X.shape[1], 10))
    W_fit = fista.fista(fun_f, W_init, params_f=lam, prox_g=None, tol=tol)

    # Check optimality conditions.
    W_grad = jax.grad(fun_f)(W_fit, lam)
    self.assertLessEqual(jnp.sqrt(jnp.sum(W_grad ** 2)), tol)

    # Compare against sklearn.
    W_skl = _logreg_skl(X, y, lam)
    self.assertArraysAllClose(W_fit, W_skl, atol=atol)

  def test_logreg_with_intercept(self):
    X, y = datasets.load_digits(return_X_y=True)
    lam = float(X.shape[0])
    tol = 1e-3
    max_iter = 200
    atol = 1e-3
    fun_f = _make_logreg_objective(X, y, fit_intercept=True)

    pytree_init = (jnp.zeros((X.shape[1], 10)), jnp.zeros(10))
    pytree_fit = fista.fista(fun_f, pytree_init, params_f=lam, prox_g=None,
                             tol=tol)

    # Check optimality conditions.
    pytree_grad = jax.grad(fun_f)(pytree_fit, lam)
    self.assertLessEqual(tu.tree_l2_norm(pytree_grad), tol)

    # Compare against sklearn.
    W_skl, b_skl = _logreg_skl(X, y, lam, fit_intercept=True)
    self.assertArraysAllClose(pytree_fit[0], W_skl, atol=atol)
    self.assertArraysAllClose(pytree_fit[1], b_skl, atol=atol)

  def test_logreg_implicit_diff(self):
    X, y = datasets.load_digits(return_X_y=True)
    lam = float(X.shape[0])
    eps = 1e-5
    fun_f = _make_logreg_objective(X, y)

    # Jacobian w.r.t. lam using finite central finite difference.
    # We use the sklearn solver for precision, as it operates on float64.
    jac_num = (_logreg_skl(X, y, lam + eps) -
               _logreg_skl(X, y, lam - eps)) / (2 * eps)

    # Compute the Jacobian w.r.t. lam (params_f) via implicit VJPs.
    W_skl = _logreg_skl(X, y, lam)
    I = jnp.eye(W_skl.size)
    I = I.reshape(-1, *W_skl.shape)
    fun = lambda g: prox_fixed_point_vjp(fun_f=fun_f, sol=W_skl, cotangent=g,
                                         params_f=lam)[0]
    jac_lam2 = jax.vmap(fun)(I).reshape(*W_skl.shape)
    self.assertArraysAllClose(jac_num, jac_lam2, atol=1e-3)

    # Make sure the decorator works.
    W_init = jnp.zeros_like(W_skl)
    tol = 1e-3
    max_iter = 200

    jac_fun = jax.jacrev(fista.fista, argnums=2)
    jac_custom = jac_fun(fun_f, W_init, lam, prox_g=None, tol=tol,
                         max_iter=max_iter, acceleration=True,
                         implicit_diff=True)
    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-2)

    # Compute the Jacobian w.r.t. lam (params_f) using JVPs.
    jac_params_f = prox_fixed_point_jvp(fun_f=fun_f, sol=W_skl, params_f=lam,
                                        tangents=(1.0, None))[0]
    self.assertArraysAllClose(jac_num, jac_params_f, atol=1e-3)

  @parameterized.product(acceleration=[True, False])
  def test_logreg_forward_diff(self, acceleration):
    X, y = datasets.load_digits(return_X_y=True)
    lam = float(X.shape[0])
    tol = 1e-3 if acceleration else 5e-3
    eps = 1e-5
    max_iter = 200
    atol = 1e-3 if acceleration else 1e-1
    fun_f = _make_logreg_objective(X, y)

    jac_lam = (_logreg_skl(X, y, lam + eps) -
               _logreg_skl(X, y, lam - eps)) / (2 * eps)

    # Compute the Jacobian w.r.t. lam via forward differentiation.
    W_init = jnp.zeros((X.shape[1], 10))
    jac_fun = jax.jacfwd(fista.fista, argnums=2)
    jac_lam2 = jac_fun(fun_f, W_init, lam, prox_g=None, tol=tol,
                       max_iter=max_iter, implicit_diff=False,
                       acceleration=acceleration)
    self.assertArraysAllClose(jac_lam, jac_lam2, atol=atol)

  def test_multiclass_svm_dual(self):
    X, y = datasets.load_digits(return_X_y=True)
    # Transform labels to a one-hot representation.
    # Y has shape (n_samples, n_classes).
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    lam = 1e5
    tol = 1e-2
    max_iter = 200
    atol = 1e-3

    def fun_f(beta, lam):
      dual_obj = jnp.vdot(beta, 1 - Y)
      V = jnp.dot(X.T, (Y - beta))
      dual_obj = dual_obj - 0.5 / lam * jnp.vdot(V, V)
      return -dual_obj  # We want to maximize the dual objective

    proj_vmap = jax.vmap(projection.projection_simplex)
    prox_g = lambda x, params_g, scaling=1.0: proj_vmap(x)

    n_samples, n_classes = Y.shape
    beta_init = jnp.ones((n_samples, n_classes)) / n_classes
    beta_fit = fista.fista(fun_f, beta_init, params_f=lam, prox_g=prox_g,
                           tol=tol)

    # Check optimality conditions.
    beta_fit2 = proj_vmap(beta_fit - jax.grad(fun_f)(beta_fit, lam))
    self.assertLessEqual(jnp.sqrt(jnp.sum((beta_fit - beta_fit2) ** 2)), tol)

    # Compare against sklearn.
    svc = svm.LinearSVC(loss="hinge", dual=True, multi_class="crammer_singer",
                        C=1.0 / lam, fit_intercept=False)
    W_skl = svc.fit(X, y).coef_.T
    W_fit = jnp.dot(X.T, (Y - beta_fit)) / lam
    self.assertArraysAllClose(W_fit, W_skl, atol=atol)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
