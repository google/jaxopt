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

"""Common objective functions."""

from typing import Tuple

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import loss


class CompositeLinearFunction:
  """A base class to represent composite linear functions.

  These are functions of the form::

    fun(params, *args, **kwargs) =
      subfun(linop(params), *args, **kwargs) + vdot(params, b(*args, **kwargs))

  where  ``linop = make_linop(*args, **kwargs)``.
  """

  def b(self, *args, **kwargs):
    """Linear term in the function."""
    return None

  def lipschitz_const(self, hyperparams):
    """Lipschitz-constant of subfun."""
    raise NotImplementedError

  def subfun(self, predictions, *args, **kwargs):
    """To be implemented by the child class."""
    raise NotImplementedError

  def __call__(self, params, *args, **kwargs):
    linop = self.make_linop(*args, **kwargs)
    predictions = linop.matvec(params)
    ret = self.subfun(predictions, *args, **kwargs)
    b = self.b(*args, **kwargs)
    if b is not None:
      ret += jnp.vdot(params, b)
    return ret


class LeastSquares(CompositeLinearFunction):
  """Least squares."""

  def subfun(self, predictions, data):
    y = data[1]
    residuals = predictions - y
    return 0.5 * jnp.mean(residuals ** 2)

  def make_linop(self, data):
    """Creates linear operator."""
    return base.LinearOperator(data[0])

  def columnwise_lipschitz_const(self, data):
    """Column-wise Lipschitz constants."""
    linop = self.make_linop(data)
    return linop.column_l2_norms(squared=True) * 1.0

  def __call__(self, W, data):
    r"""Least squares.

    .. math::

      \frac{1}{2n} ||XW - y||_2^2

    Args:
      W: parameters.
      data: a tuple ``(X, y)`` where ``X`` is a matrix of shape ``(n_samples,
        n_features)`` and ``y`` is a vector of shape ``(n_samples,)``.
    Returns:
      objective value.

    Example::

      value = least_squares(W, (X, y))
    """
    return super().__call__(W, data)


least_squares = LeastSquares()
least_squares.__doc__ = least_squares.__call__.__doc__


def ridge_regression(
  params: jnp.ndarray,
  l2reg: float,
  data: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
  r"""
  Ridge regression, i.e L2-regularized least squares.

  .. math::

    \frac{1}{2n} ||XW - y||_2^2 +
    0.5 \cdot \text{l2reg} \cdot ||W||_2^2

  Args:
    W: parameters.
    l2reg: strenght of regularization.
    data: a tuple ``(X, y)`` where ``X`` is a matrix of shape ``(n_samples,
      n_features)`` and ``y`` is a vector of shape ``(n_samples,)``.
  Returns:
    objective value.

  Example::

    value = ridge_regression(W, l2reg, (X, y))
  """
  least_squares = LeastSquares()(params, data)
  ridge = 0.5 * l2reg * jnp.sum(params ** 2)
  return least_squares + ridge


_logloss_vmap = jax.vmap(loss.multiclass_logistic_loss)


class MulticlassLogreg(CompositeLinearFunction):
  """Multiclass logistic regression objective."""

  def subfun(self, predictions, data):
    y = data[1]
    return jnp.mean(_logloss_vmap(y, predictions))

  def make_linop(self, data):
    """Creates linear operator."""
    return base.LinearOperator(data[0])

  def columnwise_lipschitz_const(self, data):
    """Column-wise Lipschitz constants."""
    linop = self.make_linop(data)
    return linop.column_l2_norms(squared=True) * 0.5

  def __call__(self, W, data):
    r"""Multiclass logistic regression.

    .. math::

      \frac{1}{n} \sum_{i=1}^n \ell(W^\top x_i, y_i)

    where :math:`\ell` is :func:`multiclass_logistic_loss
    <jaxopt.loss.multiclass_logistic_loss>` and ``X, y = data``.

    Args:
      W: a matrix of shape ``(n_features, n_classes)``.
      data: a tuple ``(X, y)`` where ``X`` is a matrix of shape ``(n_samples,
        n_features)`` and ``y`` is a vector of shape ``(n_samples,)``.
    Returns:
      objective value.

    Example::

      value = multiclass_logreg(W, (X, y))
    """
    return super().__call__(W, data)


multiclass_logreg = MulticlassLogreg()
multiclass_logreg.__doc__ = multiclass_logreg.__call__.__doc__


def multiclass_logreg_with_intercept(
  params: Tuple[jnp.ndarray, jnp.ndarray],
  data: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
  r"""
  Multiclass logistic regression with intercept.

  .. math::

    \frac{1}{n} \sum_{i=1}^n \ell(W^\top x_i + b, y_i)

  where :math:`\ell` is :func:`multiclass_logistic_loss
  <jaxopt.loss.multiclass_logistic_loss>`, ``W, b = params`` and
  ``X, y = data``.

  Args:
    params: a tuple ``(W, b)``, where ``W`` is a matrix of shape ``(n_features,
      n_classes)`` and ``b`` is a vector of shape ``(n_classes,)``.
    data: a tuple ``(X, y)`` where ``X`` is a matrix of shape ``(n_samples,
      n_features)`` and ``y`` is a vector of shape ``(n_samples,)``.
  Returns:
    objective value.
  """
  X, y = data
  W, b = params
  y_pred = jnp.dot(X, W) + b
  return jnp.mean(_logloss_vmap(y, y_pred))


def l2_multiclass_logreg(W: jnp.ndarray,
                         l2reg: float,
                         data: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
  r"""
  L2-regularized multiclass logistic regression.

  .. math::

    \frac{1}{n} \sum_{i=1}^n \ell(W^\top x_i, y_i) +
    0.5 \cdot \text{l2reg} \cdot ||W||_2^2

  where :math:`\ell` is :func:`multiclass_logistic_loss
  <jaxopt.loss.multiclass_logistic_loss>` and ``X, y = data``.

  Args:
    W: a matrix of shape ``(n_features, n_classes)``.
    data: a tuple ``(X, y)`` where ``X`` is a matrix of shape ``(n_samples,
      n_features)`` and ``y`` is a vector of shape ``(n_samples,)``.
  Returns:
    objective value.
  """
  X, y = data
  y_pred = jnp.dot(X, W)
  return jnp.mean(_logloss_vmap(y, y_pred)) + 0.5 * l2reg * jnp.sum(W ** 2)


def l2_multiclass_logreg_with_intercept(
  params: Tuple[jnp.ndarray, jnp.ndarray],
  l2reg: float,
  data: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
  r"""
  L2-regularized multiclass logistic regression with intercept.

  .. math::

    \frac{1}{n} \sum_{i=1}^n \ell(W^\top x_i + b, y_i) +
    0.5 \cdot \text{l2reg} \cdot ||W||_2^2

  where :math:`\ell` is :func:`multiclass_logistic_loss
  <jaxopt.loss.multiclass_logistic_loss>`, ``W, b = params`` and
  ``X, y = data``.

  Args:
    params: a tuple ``(W, b)``, where ``W`` is a matrix of shape ``(n_features,
      n_classes)`` and ``b`` is a vector of shape ``(n_classes,)``.
    data: a tuple ``(X, y)`` where ``X`` is a matrix of shape ``(n_samples,
      n_features)`` and ``y`` is a vector of shape ``(n_samples,)``.
  Returns:
    objective value.
  """
  X, y = data
  W, b = params
  y_pred = jnp.dot(X, W) + b
  return jnp.mean(_logloss_vmap(y, y_pred)) + 0.5 * l2reg * jnp.sum(W ** 2)


_binary_logloss_vmap = jax.vmap(loss.binary_logistic_loss)


class BinaryLogreg(CompositeLinearFunction):
  """Binary logistic regression objective."""

  def subfun(self, predictions, data):
    y = data[1]
    return jnp.mean(_binary_logloss_vmap(y, predictions))

  def make_linop(self, data):
    """Creates linear operator."""
    return base.LinearOperator(data[0])

  def columnwise_lipschitz_const(self, data):
    """Column-wise Lipschitz constants."""
    linop = self.make_linop(data)
    return linop.column_l2_norms(squared=True) * 0.25

  def __call__(self, w, data):
    r"""Binary logistic regression.

    .. math::

      \frac{1}{n} \sum_{i=1}^n \ell(w^\top x_i, y_i)

    where :math:`\ell` is :func:`binary_logistic_loss
    <jaxopt.loss.binary_logistic_loss>` and ``X, y = data``.

    Args:
      w: a vector of shape ``(n_features, )``.
      data: a tuple ``(X, y)`` where ``X`` is a matrix of shape ``(n_samples,
        n_features)`` and ``y`` is a vector of shape ``(n_samples,)``,
        containing ``0`` or ``1`` values.
    Returns:
      objective value.

    Example::

      value = binary_logreg(w, (X, y))
    """
    return super().__call__(w, data)


binary_logreg = BinaryLogreg()
binary_logreg.__doc__ = binary_logreg.__call__.__doc__


class MulticlassLinearSvmDual(CompositeLinearFunction):
  """Dual objective function of multiclass linear SVMs."""

  def subfun(self, Xbeta, l2reg, data):
    X, Y = data
    XY = jnp.dot(X.T, Y)  # todo: avoid storing / computing this matrix.

    # The dual objective is:
    # fun(beta) = vdot(beta, 1 - Y) - 0.5 / l2reg * ||V(beta)||^2
    # where V(beta) = dot(X.T, Y) - dot(X.T, beta).
    V = XY - Xbeta
    # With opposite sign, as we want to maximize.
    return 0.5 / l2reg * jnp.vdot(V, V)

  def make_linop(self, l2reg, data):
    """Creates linear operator."""
    return base.LinearOperator(data[0].T)

  def columnwise_lipschitz_const(self, l2reg, data):
    """Column-wise Lipschitz constants."""
    linop = self.make_linop(l2reg, data)
    return linop.column_l2_norms(squared=True)

  def b(self, l2reg, data):
    return data[1] - 1


multiclass_linear_svm_dual = MulticlassLinearSvmDual()
