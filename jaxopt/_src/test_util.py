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

from absl.testing import absltest
from absl.testing import parameterized

import functools

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


# Test utilities copied from JAX core so we don't depend on their private API.

_dtype_to_32bit_dtype = {
    onp.dtype('int64'): onp.dtype('int32'),
    onp.dtype('uint64'): onp.dtype('uint32'),
    onp.dtype('float64'): onp.dtype('float32'),
    onp.dtype('complex128'): onp.dtype('complex64'),
}

@functools.lru_cache(maxsize=None)
def _canonicalize_dtype(x64_enabled, dtype):
  """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
  try:
    dtype = onp.dtype(dtype)
  except TypeError as e:
    raise TypeError(f'dtype {dtype!r} not understood') from e

  if x64_enabled:
    return dtype
  else:
    return _dtype_to_32bit_dtype.get(dtype, dtype)


def canonicalize_dtype(dtype):
  return _canonicalize_dtype(jax.config.x64_enabled, dtype)


python_scalar_dtypes : dict = {
  bool: onp.dtype('bool'),
  int: onp.dtype('int64'),
  float: onp.dtype('float64'),
  complex: onp.dtype('complex128'),
}


def _dtype(x):
  return (getattr(x, 'dtype', None) or
          onp.dtype(python_scalar_dtypes.get(type(x), None)) or
          onp.asarray(x).dtype)


float0: onp.dtype = onp.dtype([('float0', onp.void, 0)])


_default_tolerance = {
  float0: 0,
  onp.dtype(onp.bool_): 0,
  onp.dtype(onp.int8): 0,
  onp.dtype(onp.int16): 0,
  onp.dtype(onp.int32): 0,
  onp.dtype(onp.int64): 0,
  onp.dtype(onp.uint8): 0,
  onp.dtype(onp.uint16): 0,
  onp.dtype(onp.uint32): 0,
  onp.dtype(onp.uint64): 0,
  #onp.dtype(_dtypes.bfloat16): 1e-2,
  onp.dtype(onp.float16): 1e-3,
  onp.dtype(onp.float32): 1e-6,
  onp.dtype(onp.float64): 1e-15,
  onp.dtype(onp.complex64): 1e-6,
  onp.dtype(onp.complex128): 1e-15,
}


def default_tolerance():
  if device_under_test() != "tpu":
    return _default_tolerance
  tol = _default_tolerance.copy()
  tol[np.dtype(np.float32)] = 1e-3
  tol[np.dtype(np.complex64)] = 1e-3
  return


def tolerance(dtype, tol=None):
  tol = {} if tol is None else tol
  if not isinstance(tol, dict):
    return tol
  tol = {onp.dtype(key): value for key, value in tol.items()}
  dtype = canonicalize_dtype(onp.dtype(dtype))
  return tol.get(dtype, default_tolerance()[dtype])


def device_under_test():
  return jax.lib.xla_bridge.get_backend().platform


def _assert_numpy_allclose(a, b, atol=None, rtol=None, err_msg=''):
  if a.dtype == b.dtype == float0:
    np.testing.assert_array_equal(a, b, err_msg=err_msg)
    return
  #a = a.astype(np.float32) if a.dtype == _dtypes.bfloat16 else a
  #b = b.astype(np.float32) if b.dtype == _dtypes.bfloat16 else b
  kw = {}
  if atol: kw["atol"] = atol
  if rtol: kw["rtol"] = rtol
  with onp.errstate(invalid='ignore'):
    # TODO(phawkins): surprisingly, assert_allclose sometimes reports invalid
    # value errors. It should not do that.
    onp.testing.assert_allclose(a, b, **kw, err_msg=err_msg)


def is_sequence(x):
  try:
    iter(x)
  except TypeError:
    return False
  else:
    return True


class JaxoptTestCase(parameterized.TestCase):
  """Base class for JAXopt tests including numerical checks."""

  def assertArraysEqual(self, x, y, *, check_dtypes=True, err_msg=''):
    """Assert that x and y arrays are exactly equal."""
    if check_dtypes:
      self.assertDtypesMatch(x, y)
    # Work around https://github.com/numpy/numpy/issues/18992
    with onp.errstate(over='ignore'):
      onp.testing.assert_array_equal(x, y, err_msg=err_msg)

  def assertArraysAllClose(self, x, y, *, check_dtypes=True, atol=None,
                           rtol=None, err_msg=''):
    """Assert that x and y are close (up to numerical tolerances)."""
    self.assertEqual(x.shape, y.shape)
    atol = max(tolerance(_dtype(x), atol), tolerance(_dtype(y), atol))
    rtol = max(tolerance(_dtype(x), rtol), tolerance(_dtype(y), rtol))

    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)

    if check_dtypes:
      self.assertDtypesMatch(x, y)

  def assertDtypesMatch(self, x, y, *, canonicalize_dtypes=True):
    if not jax.config.x64_enabled and canonicalize_dtypes:
      self.assertEqual(canonicalize_dtype(_dtype(x)),
                       canonicalize_dtype(_dtype(y)))
    else:
      self.assertEqual(_dtype(x), _dtype(y))

  def assertAllClose(self, x, y, *, check_dtypes=True, atol=None, rtol=None,
                     canonicalize_dtypes=True, err_msg=''):
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict, msg=err_msg)
      self.assertEqual(set(x.keys()), set(y.keys()), msg=err_msg)
      for k in x.keys():
        self.assertAllClose(x[k], y[k], check_dtypes=check_dtypes, atol=atol,
                            rtol=rtol, canonicalize_dtypes=canonicalize_dtypes,
                            err_msg=err_msg)
    elif is_sequence(x) and not hasattr(x, '__array__'):
      self.assertTrue(
          is_sequence(y) and not hasattr(y, '__array__'), msg=err_msg
      )
      self.assertEqual(len(x), len(y), msg=err_msg)
      for x_elt, y_elt in zip(x, y):
        self.assertAllClose(x_elt, y_elt, check_dtypes=check_dtypes, atol=atol,
                            rtol=rtol, canonicalize_dtypes=canonicalize_dtypes,
                            err_msg=err_msg)
    elif hasattr(x, '__array__') or onp.isscalar(x):
      self.assertTrue(
          hasattr(y, '__array__') or onp.isscalar(y),
          msg=f'{err_msg}: {x} is an array but {y} is not.',
      )
      if check_dtypes:
        self.assertDtypesMatch(x, y, canonicalize_dtypes=canonicalize_dtypes)
      x = onp.asarray(x)
      y = onp.asarray(y)
      self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
                                err_msg=err_msg)
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))
