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

from jaxopt import loss

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm


def make_logreg_objective(X, y, fit_intercept=False):
  X = preprocessing.Normalizer().fit_transform(X)
  if fit_intercept:
    def fun(pytree, lam):
      W, b = pytree
      logits = jnp.dot(X, W) + b
      return (jnp.sum(jax.vmap(loss.multiclass_logistic_loss)(y, logits)) +
              0.5 * lam * jnp.sum(W ** 2))
  else:
    def fun(W, lam):
      logits = jnp.dot(X, W)
      return (jnp.sum(jax.vmap(loss.multiclass_logistic_loss)(y, logits)) +
              0.5 * lam * jnp.sum(W ** 2))
  return fun


def logreg_skl(X, y, lam, tol=1e-5, fit_intercept=False):
  """Return the solution found by sklearn's logreg solver."""
  X = preprocessing.Normalizer().fit_transform(X)
  logreg = linear_model.LogisticRegression(fit_intercept=fit_intercept,
                                           C=1. / lam, tol=tol,
                                           multi_class="multinomial")
  logreg = logreg.fit(X, y)
  if fit_intercept:
    return logreg.coef_.T, logreg.intercept_
  else:
    return logreg.coef_.T


def logreg_skl_jac(X, y, lam, tol=1e-5, fit_intercept=False, eps=1e-5):
  """Return the numerical Jacobian using sklearn's logreg solver."""
  res1 = logreg_skl(X, y, lam + eps, tol, fit_intercept)
  res2 = logreg_skl(X, y, lam - eps, tol, fit_intercept)
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


def make_least_squares_objective(X, y, fit_intercept=False):
  X = preprocessing.Normalizer().fit_transform(X)
  if fit_intercept:
    def fun(pytree, lam):
      w, b = pytree
      y_pred = jnp.dot(X, w) + b
      diff = y_pred - y
      return 0.5 / (lam * X.shape[0]) * jnp.dot(diff, diff)
  else:
    def fun(w, lam):
      y_pred = jnp.dot(X, w)
      diff = y_pred - y
      return 0.5 / (lam * X.shape[0]) * jnp.dot(diff, diff)
  return fun


def lasso_skl_jac(X, y, lam, tol=1e-5, fit_intercept=False, eps=1e-5):
  """Return the numerical Jacobian using sklearn's lasso solver."""
  res1 = lasso_skl(X, y, lam + eps, tol, fit_intercept)
  res2 = lasso_skl(X, y, lam - eps, tol, fit_intercept)
  twoeps = 2 * eps
  if fit_intercept:
    return (res1[0] - res2[0]) / twoeps, (res1[1] - res2[1]) / twoeps
  else:
    return (res1 - res2) / twoeps


def enet_skl(X, y, lam, gamma, tol=1e-5, fit_intercept=False):
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


def enet_skl_jac(X, y, lam, gamma, tol=1e-5, fit_intercept=False, eps=1e-5):
  if fit_intercept:
    raise NotImplementedError

  jac_lam = (enet_skl(X, y, lam + eps, gamma) -
             enet_skl(X, y, lam - eps, gamma)) / (2 * eps)
  jac_gam = (enet_skl(X, y, lam, gamma + eps) -
             enet_skl(X, y, lam, gamma - eps)) / (2 * eps)

  return jac_lam, jac_gam


def make_multiclass_linear_svm_objective(X, y):
  Y = preprocessing.LabelBinarizer().fit_transform(y)
  def fun(beta, lam):
    dual_obj = jnp.vdot(beta, 1 - Y)
    V = jnp.dot(X.T, (Y - beta))
    dual_obj = dual_obj - 0.5 / lam * jnp.vdot(V, V)
    return -dual_obj  # We want to maximize the dual objective
  return fun


def multiclass_linear_svm_skl(X, y, lam, tol=1e-5):
  svc = svm.LinearSVC(loss="hinge", dual=True, multi_class="crammer_singer",
                      C=1.0 / lam, fit_intercept=False, tol=tol).fit(X, y)
  return svc.coef_.T


def multiclass_linear_svm_skl_jac(X, y, lam, tol=1e-5, eps=1e-5):
  return (multiclass_linear_svm_skl(X, y, lam + eps, tol=tol) -
          multiclass_linear_svm_skl(X, y, lam - eps, tol=tol)) / (2 * eps)
