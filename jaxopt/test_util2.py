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

from sklearn import linear_model
from sklearn import svm


class LeastSquaresFunction(base.CompositeLinearFunction2):

  def subfun(self, predictions, hyperparams, data):
    del hyperparams  # not used
    y = data[1]
    residuals = predictions - y
    return 0.5 * jnp.mean(residuals ** 2)

  def lipschitz_const(self, hyperparams):
    return 1.0


least_squares_objective = LeastSquaresFunction()


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


def enet_skl(X, y, params_prox, tol=1e-5, fit_intercept=False):
  lam, gamma = params_prox
  alpha = lam + lam * gamma
  l1_ratio = lam / alpha
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


def multitask_lasso_skl(X, Y, lam, tol=1e-5):
  """Return the solution found by sklearn's multitask lasso solver."""
  lasso = linear_model.MultiTaskLasso(fit_intercept=False, alpha=lam, tol=tol)
  lasso = lasso.fit(X, Y)
  return lasso.coef_.T


_logloss_vmap = jax.vmap(loss.multiclass_logistic_loss)


class MulticlassLogregFunction(base.CompositeLinearFunction2):

  def subfun(self, predictions, hyperparams, data):
    del hyperparams  # not used
    y = data[1]
    return jnp.mean(_logloss_vmap(y, predictions))

  def lipschitz_const(self, hyperparams):
    return 0.5


multiclass_logreg_objective = MulticlassLogregFunction()


_binary_logloss_vmap = jax.vmap(loss.binary_logistic_loss)


class BinaryLogregFunction(base.CompositeLinearFunction2):

  def subfun(self, predictions, hyperparams, data):
    del hyperparams  # not used
    y = data[1]
    return jnp.mean(_binary_logloss_vmap(y, predictions))

  def lipschitz_const(self, hyperparams):
    return 0.25


binary_logreg_objective = BinaryLogregFunction()


def l2_logreg_objective(params, hyperparams, data):
  X, y = data
  y_pred = jnp.dot(X, params)
  ret = jnp.mean(_logloss_vmap(y, y_pred))
  ret += 0.5 * hyperparams * jnp.sum(params ** 2)
  return ret


def l2_logreg_objective_with_intercept(params, hyperparams, data):
  X, y = data
  W, b = params
  y_pred = jnp.dot(X, W) + b
  ret = jnp.mean(_logloss_vmap(y, y_pred))
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
                                           C=1. / (lam * X.shape[0]), tol=tol,
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


class MulticlassLinearSvmDual(base.CompositeLinearFunction2):
  """Dual objective function of multiclass linear SVMs."""

  def subfun(self, Xbeta, lam, data):
    X, Y = data
    XY = jnp.dot(X.T, Y)

    # The dual objective is:
    # fun(beta) = vdot(beta, 1 - Y) - 0.5 / lam * ||V(beta)||^2
    # where V(beta) = dot(X.T, Y) - dot(X.T, beta).

    V = XY - Xbeta
    # With opposite sign, as we want to maximize.
    return 0.5 / lam * jnp.vdot(V, V)

  def b(self, data):
    return data[1] - 1

  def make_linop(self, data):
    return base.LinearOperator(data[0].T)

  def lipschitz_const(self, hyperparams):
    return 1.0

multiclass_linear_svm_dual_objective = MulticlassLinearSvmDual()


def multiclass_linear_svm_skl(X, y, lam, tol=1e-5):
  svc = svm.LinearSVC(loss="hinge", dual=True, multi_class="crammer_singer",
                      C=1.0 / lam, fit_intercept=False, tol=tol).fit(X, y)
  return svc.coef_.T
