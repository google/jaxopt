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

"""Test utilities."""

import jax
import jax.numpy as jnp

from jaxopt import base
from jaxopt import loss

from sklearn import datasets
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm


def _make_composite_linear_objective(loss_fun, lipschitz_constant, X, y,
                                     fit_intercept=False, l2_penalty=True,
                                     preprocess_X=True):
  if preprocess_X:
    X = preprocessing.Normalizer().fit_transform(X)

  loss_vmap = jax.vmap(loss_fun)

  if fit_intercept:
    def fun(pytree, lam):
      W, b = pytree
      y_pred = jnp.dot(X, W) + b
      ret = jnp.sum(loss_vmap(y, y_pred))
      if l2_penalty:
        ret += 0.5 * lam * jnp.sum(W ** 2)
      return ret

    return fun

  else:
    if l2_penalty:
      def fun(W, lam):
        y_pred = jnp.dot(X, W)
        return jnp.sum(loss_vmap(y, y_pred)) + 0.5 * lam * jnp.sum(W ** 2)
      return fun
    else:
      def subfun(y_pred, lam):
        return jnp.sum(loss_vmap(y, y_pred))

      lipschitz_fun = lambda params_fun: lipschitz_constant
      linop = base.LinearOperator(X)
      return base.CompositeLinearFunction(subfun, linop,
                                          lipschitz_fun=lipschitz_fun)


def make_logreg_objective(X, y, fit_intercept=False, l2_penalty=True,
                          preprocess_X=True):
  loss_fun = loss.multiclass_logistic_loss
  return _make_composite_linear_objective(loss_fun=loss_fun,
                                          lipschitz_constant=0.5, X=X, y=y,
                                          fit_intercept=fit_intercept,
                                          l2_penalty=l2_penalty,
                                          preprocess_X=preprocess_X)


def make_binary_logreg_objective(X, y, fit_intercept=False, l2_penalty=True,
                                 preprocess_X=True):
  loss_fun = loss.binary_logistic_loss
  return _make_composite_linear_objective(loss_fun=loss_fun,
                                          lipschitz_constant=0.25, X=X, y=y,
                                          fit_intercept=fit_intercept,
                                          l2_penalty=l2_penalty,
                                          preprocess_X=preprocess_X)


def logreg_skl(X, y, lam, tol=1e-5, fit_intercept=False,
               penalty="l2", multiclass=True):
  """Return the solution found by sklearn's logreg solver."""

  def _reshape_coef(coef):
    return coef.ravel() if coef.shape[0] == 1 else coef.T

  X = preprocessing.Normalizer().fit_transform(X)
  if multiclass:
    solver = "lbfgs"
    multiclass = "multinomial"
  else:
    solver = "liblinear"
    multiclass = "ovr"
  logreg = linear_model.LogisticRegression(fit_intercept=fit_intercept,
                                           C=1. / lam, tol=tol,
                                           solver=solver, penalty=penalty,
                                           multi_class=multiclass,
                                           random_state=0)
  logreg = logreg.fit(X, y)
  if fit_intercept:
    return _reshape_coef(logreg.coef_), logreg.intercept_
  else:
    return _reshape_coef(logreg.coef_)


def logreg_skl_jac(X, y, lam, tol=1e-5, fit_intercept=False, penalty="l2",
                   multiclass=True, eps=1e-5):
  """Return the numerical Jacobian using sklearn's logreg solver."""
  res1 = logreg_skl(X, y, lam + eps, tol, fit_intercept, penalty, multiclass)
  res2 = logreg_skl(X, y, lam - eps, tol, fit_intercept, penalty, multiclass)
  twoeps = 2 * eps
  if fit_intercept:
    return (res1[0] - res2[0]) / twoeps, (res1[1] - res2[1]) / twoeps
  else:
    return (res1 - res2) / twoeps


def lasso_skl(X, y, lam, tol=1e-5, fit_intercept=False):
  """Return the solution found by sklearn's lasso solver."""
  X = preprocessing.Normalizer().fit_transform(X)
  lasso = linear_model.Lasso(fit_intercept=fit_intercept, alpha=lam, tol=tol)
  lasso = lasso.fit(X, y)
  if fit_intercept:
    return lasso.coef_, lasso.intercept_
  else:
    return lasso.coef_


def multitask_lasso_skl(X, Y, lam, tol=1e-5, fit_intercept=False):
  """Return the solution found by sklearn's multitask lasso solver."""
  X = preprocessing.Normalizer().fit_transform(X)
  lasso = linear_model.MultiTaskLasso(fit_intercept=False, alpha=lam, tol=tol)
  lasso = lasso.fit(X, Y)
  return lasso.coef_.T


def make_least_squares_objective(X, y, fit_intercept=False, preprocess_X=True):
  """Constructs a least squares loss objective function.

  If `fit_intercept=False`, the returned callable takes the form::

      fun(w, lam) = (0.5 / (lam * n_features)) || X w - y||^2

  where ``w`` is an array, ``lam`` is a scalar and ``n_features = X.shape[1]`.

  If `fit_intercept=True`, the returned callable takes the form::

      fun((w, b), lam) = (0.5 / (lam * n_features)) || X w + b - y||^2.

  Args:
    X: input dataset, of size (n_samples, n_features)
    y: target values associated with X, same shape as ``dot(X, w)``.
    fit_intercept: whether to fit an intercept or not, see above.
    preprocess_X: whether to normalize X using sklearn.preprocessing.Normalizer
      or not.
  Returns:
    fun: least squares loss (see above).
  """
  if preprocess_X:
    X = preprocessing.Normalizer().fit_transform(X)
  if fit_intercept:
    def fun(pytree, lam):
      w, b = pytree
      y_pred = jnp.dot(X, w) + b
      diff = y_pred - y
      return 0.5 / (lam * X.shape[0]) * jnp.dot(diff, diff)
    return fun
  else:
    def subfun(y_pred, lam):
      diff = y_pred - y
      return 0.5 / (lam * y_pred.shape[0]) * jnp.vdot(diff, diff)
    linop = base.LinearOperator(X)
    return base.CompositeLinearFunction(subfun, linop)


def lasso_skl_jac(X, y, lam, tol=1e-5, fit_intercept=False, eps=1e-5):
  """Return the numerical Jacobian using sklearn's lasso solver."""
  res1 = lasso_skl(X, y, lam + eps, tol, fit_intercept)
  res2 = lasso_skl(X, y, lam - eps, tol, fit_intercept)
  twoeps = 2 * eps
  if fit_intercept:
    return (res1[0] - res2[0]) / twoeps, (res1[1] - res2[1]) / twoeps
  else:
    return (res1 - res2) / twoeps


def enet_skl(X, y, params_prox, tol=1e-5, fit_intercept=False):
  lam, gamma = params_prox
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


def enet_skl_jac(X, y, params_prox, tol=1e-5, fit_intercept=False, eps=1e-5):
  if fit_intercept:
    raise NotImplementedError

  lam, gamma = params_prox

  jac_lam = (enet_skl(X, y, (lam + eps, gamma)) -
             enet_skl(X, y, (lam - eps, gamma))) / (2 * eps)
  jac_gam = (enet_skl(X, y, (lam, gamma + eps)) -
             enet_skl(X, y, (lam, gamma - eps))) / (2 * eps)

  return jac_lam, jac_gam


def make_multiclass_linear_svm_objective(X, y):
  Y = preprocessing.LabelBinarizer().fit_transform(y)
  XY = jnp.dot(X.T, Y)

  # The dual objective is:
  # fun(beta) = vdot(beta, 1 - Y) - 0.5 / lam * ||V(beta)||^2
  # where V(beta) = dot(X.T, Y) - dot(X.T, beta).
  def subfun(Xbeta, lam):
    V = XY - Xbeta
    # With opposite sign, as we want to maximize.
    return 0.5 / lam * jnp.vdot(V, V)

  linop = base.LinearOperator(X.T)
  # With opposite sign, as we want to maximize.
  b = Y - 1
  return base.CompositeLinearFunction(subfun, linop, b)


def multiclass_linear_svm_skl(X, y, lam, tol=1e-5):
  svc = svm.LinearSVC(loss="hinge", dual=True, multi_class="crammer_singer",
                      C=1.0 / lam, fit_intercept=False, tol=tol).fit(X, y)
  return svc.coef_.T


def multiclass_linear_svm_skl_jac(X, y, lam, tol=1e-5, eps=1e-5):
  return (multiclass_linear_svm_skl(X, y, lam + eps, tol=tol) -
          multiclass_linear_svm_skl(X, y, lam - eps, tol=tol)) / (2 * eps)


def test_logreg_vmap(make_solver_fun, make_fixed_point_fun, params_list,
                     l2_penalty=True, unpack_params=False):
  X, y = datasets.make_classification(n_samples=30, n_features=5,
                                      n_informative=3, n_classes=2,
                                      random_state=0)
  fun = make_logreg_objective(X, y, l2_penalty=l2_penalty, preprocess_X=True)
  W_init = jnp.zeros((X.shape[1], 2))
  solver_fun = make_solver_fun(fun=fun, init=W_init, tol=1e-3, maxiter=100)
  fixed_point_fun = make_fixed_point_fun(fun=fun)

  def solve(params):
    if unpack_params:
      W_fit = solver_fun(*params)
    else:
      W_fit = solver_fun(params)
    W_fit2 = fixed_point_fun(W_fit, params)
    error = jnp.sqrt(jnp.sum((W_fit - W_fit2) ** 2))
    return error

  errors = jnp.array([solve(params) for params in params_list])
  errors_vmap = jax.vmap(solve)(params_list)
  return errors, errors_vmap
