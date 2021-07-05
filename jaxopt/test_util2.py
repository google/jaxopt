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


def least_squares_objective(params, hyperparams, data):
  X, y = data
  residuals = jnp.dot(X, params) - y
  return 0.5 / len(y) * jnp.sum(residuals ** 2)


def ridge_objective(params, hyperparams, data):
  X, y = data
  residuals = jnp.dot(X, params) - y
  return (0.5 / len(y) * jnp.sum(residuals ** 2) +
          0.5 * hyperparams * jnp.sum(params ** 2))


def ridge_solver(X, y, lam):
  XX = jnp.dot(X.T, X)
  Xy = jnp.dot(X.T, y)
  I = jnp.eye(X.shape[1])
  return jnp.linalg.solve(XX + lam * len(y) * I, Xy)


def ridge_solver_jac(X, y, lam, eps=1e-4):
  return (ridge_solver(X, y, lam + eps) -
          ridge_solver(X, y, lam - eps)) / (2 * eps)


def lasso_skl(X, y, lam, tol=1e-5, fit_intercept=False):
  """Return the solution found by sklearn's lasso solver."""
  lasso = linear_model.Lasso(fit_intercept=fit_intercept, alpha=lam, tol=tol)
  lasso = lasso.fit(X, y)
  if fit_intercept:
    return lasso.coef_, lasso.intercept_
  else:
    return lasso.coef_


def lasso_skl_jac(X, y, lam, tol=1e-5, fit_intercept=False, eps=1e-4):
  """Return the numerical Jacobian using sklearn's lasso solver."""
  res1 = lasso_skl(X, y, lam + eps, tol, fit_intercept)
  res2 = lasso_skl(X, y, lam - eps, tol, fit_intercept)
  twoeps = 2 * eps
  if fit_intercept:
    return (res1[0] - res2[0]) / twoeps, (res1[1] - res2[1]) / twoeps
  else:
    return (res1 - res2) / twoeps


logloss_vmap = jax.vmap(loss.multiclass_logistic_loss)


def l2_logreg_objective(params, hyperparams, data):
  X, y = data
  y_pred = jnp.dot(X, params)
  ret = jnp.mean(logloss_vmap(y, y_pred))
  ret += 0.5 * hyperparams * jnp.sum(params ** 2)
  return ret


def l2_logreg_objective_with_intercept(params, hyperparams, data):
  X, y = data
  W, b = params
  y_pred = jnp.dot(X, W) + b
  ret = jnp.mean(logloss_vmap(y, y_pred))
  ret += 0.5 * hyperparams * jnp.sum(W ** 2)
  return ret


def logreg_skl(X, y, lam, tol=1e-5, fit_intercept=False,
               penalty="l2", multiclass=True):
  """Return the solution found by sklearn's logreg solver."""

  def _reshape_coef(coef):
    return coef.ravel() if coef.shape[0] == 1 else coef.T

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
