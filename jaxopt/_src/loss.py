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

"""Loss functions."""

import jax
from jax.nn import softplus
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxopt._src.projection import projection_simplex


# Regression


def huber_loss(target: float, pred: float, delta: float = 1.0) -> float:
  """Huber loss.

  Args:
    target: ground truth
    pred: predictions
    delta: radius of quadratic behavior
  Returns:
    loss value

  References:
    https://en.wikipedia.org/wiki/Huber_loss
  """
  abs_diff = jnp.abs(target - pred)
  return jnp.where(abs_diff > delta,
                   delta * (abs_diff - .5 * delta),
                   0.5 * abs_diff ** 2)

# Binary classification.


def binary_logistic_loss(label: int, logit: float) -> float:
  """Binary logistic loss.

  Args:
    label: ground-truth integer label (0 or 1).
    logit: score produced by the model (float).
  Returns:
    loss value
  """
  # Softplus is the Fenchel conjugate of the Fermi-Dirac negentropy on [0, 1].
  # softplus = proba * logit - xlogx(proba) - xlogx(1 - proba),
  # where xlogx(proba) = proba * log(proba).
  return softplus(logit) - label * logit


# Multiclass classification.


def multiclass_logistic_loss(label: int, logits: jnp.ndarray) -> float:
  """Multiclass logistic loss.

  Args:
    label: ground-truth integer label, between 0 and n_classes - 1.
    logits: scores produced by the model, shape = (n_classes, ).
  Returns:
    loss value
  """
  n_classes = logits.shape[0]
  one_hot = jax.nn.one_hot(label, n_classes)
  # Logsumexp is the Fenchel conjugate of the Shannon negentropy on the simplex.
  # logsumexp = jnp.dot(proba, logits) - jnp.dot(proba, jnp.log(proba))
  return logsumexp(logits) - jnp.dot(logits, one_hot)


def multiclass_sparsemax_loss(label: int, scores: jnp.ndarray) -> float:
  """Multiclass sparsemax loss.

  Args:
    label: ground-truth integer label, between 0 and n_classes - 1.
    scores: scores produced by the model, shape = (n_classes, ).
  Returns:
    loss value

  References:
    From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label
    Classification. André F. T. Martins, Ramón Fernandez Astudillo.
    ICML 2016.
  """
  n_classes = scores.shape[0]
  one_hot = jax.nn.one_hot(label, n_classes)
  proba = projection_simplex(scores)
  # Fenchel conjugate of the Gini negentropy, defined by:
  # cumulant = jnp.dot(proba, scores) + 0.5 * jnp.dot(proba, (1 - proba)).
  cumulant = 0.5 + jnp.dot(proba , scores - 0.5 * proba)
  return cumulant - jnp.dot(scores, one_hot)
