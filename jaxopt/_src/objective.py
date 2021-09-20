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


class LeastSquaresFunction(CompositeLinearFunction):
  """Least squares objective class."""

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


least_squares = LeastSquaresFunction()


_logloss_vmap = jax.vmap(loss.multiclass_logistic_loss)


class MulticlassLogregFunction(CompositeLinearFunction):
  """Multiclass logistic regression objective class."""

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


multiclass_logreg = MulticlassLogregFunction()


def multiclass_logreg_with_intercept(params, data):
  X, y = data
  W, b = params
  y_pred = jnp.dot(X, W) + b
  return jnp.mean(_logloss_vmap(y, y_pred))


def l2_multiclass_logreg(W, l2reg, data):
  X, y = data
  y_pred = jnp.dot(X, W)
  return jnp.mean(_logloss_vmap(y, y_pred)) + 0.5 * l2reg * jnp.sum(W ** 2)


def l2_multiclass_logreg_with_intercept(params, l2reg, data):
  X, y = data
  W, b = params
  y_pred = jnp.dot(X, W) + b
  return jnp.mean(_logloss_vmap(y, y_pred)) + 0.5 * l2reg * jnp.sum(W ** 2)


_binary_logloss_vmap = jax.vmap(loss.binary_logistic_loss)


class BinaryLogregFunction(CompositeLinearFunction):
  """Binary logistic regression objective class."""

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


binary_logreg = BinaryLogregFunction()


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
