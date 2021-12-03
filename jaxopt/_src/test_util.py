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

from jaxopt._src import base
from jaxopt._src import loss

import numpy as onp
import scipy as osp

from sklearn import linear_model
from sklearn import svm


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


def multiclass_linear_svm_skl(X, y, lam, tol=1e-5):
  svc = svm.LinearSVC(loss="hinge", dual=True, multi_class="crammer_singer",
                      C=1.0 / lam, fit_intercept=False, tol=tol).fit(X, y)
  return svc.coef_.T


def multiclass_linear_svm_skl_jac(X, y, lam, tol=1e-5, eps=1e-5):
  return (multiclass_linear_svm_skl(X, y, lam + eps, tol=tol) -
          multiclass_linear_svm_skl(X, y, lam - eps, tol=tol)) / (2 * eps)


def lsq_linear_osp(X, y, bounds, tol=1e-10, max_iter=None):
  return osp.optimize.lsq_linear(X, y, bounds, tol=tol, max_iter=max_iter).x


def lsq_linear_cube_osp(X, y, l, tol=1e-10, max_iter=None):
  bounds = (onp.zeros(X.shape[1]), l * onp.ones(X.shape[1]))
  return lsq_linear_osp(X, y, bounds, tol, max_iter)


def lsq_linear_cube_osp_jac(X, y, l, eps=1e-5, tol=1e-10, max_iter=None):
  return (lsq_linear_cube_osp(X, y, l + eps, tol, max_iter) -
          lsq_linear_cube_osp(X, y, l - eps, tol, max_iter)) / (2 * eps)


def check_states_have_same_types(state1, state2):
  if len(state1._fields) != len(state2._fields):
    raise ValueError("state1 and state2 should have the same number of "
                     "attributes.")

  for attr1, attr2 in zip(state1._fields, state2._fields):
    if attr1 != attr2:
      raise ValueError("Attribute names do not agree: %s and %s." % (attr1,
                                                                     attr2))

    type1 = type(getattr(state1, attr1)).__name__
    type2 = type(getattr(state2, attr2)).__name__

    if type1 != type2:
      raise ValueError("Attribute '%s' has different types in state1 and "
                       "state2: %s vs. %s" % (attr1, type1, type2))
