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

from typing import Callable

import jax
import jax.numpy as jnp
from jaxopt._src.projection import projection_simplex

from optax import losses as optax_losses


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
  return optax_losses.huber_loss(pred, target, delta)

# Binary classification.


def binary_logistic_loss(label: int, logit: float) -> float:
  """Binary logistic loss.

  Args:
    label: ground-truth integer label (0 or 1).
    logit: score produced by the model (float).
  Returns:
    loss value
  """
  return optax_losses.sigmoid_binary_cross_entropy(
      jnp.asarray(logit), jnp.asarray(label))


def binary_sparsemax_loss(label: int, logit: float) -> float:
  """Binary sparsemax loss.

  Args:
    label: ground-truth integer label (0 or 1).
    logit: score produced by the model (float).
  Returns:
    loss value

  References:
    Learning with Fenchel-Young Losses. Mathieu Blondel, André F. T. Martins,
    Vlad Niculae. JMLR 2020. (Sec. 4.4)
  """
  return optax_losses.sparsemax_loss(
      jnp.asarray(logit), jnp.asarray(label))


def binary_hinge_loss(label: int, score: float) -> float:
  """Binary hinge loss.

  Args:
    label: ground-truth integer label (0 or 1).
    score: score produced by the model (float).
  Returns:
    loss value.

  References:
    https://en.wikipedia.org/wiki/Hinge_loss
  """
  return optax_losses.hinge_loss(score, 2.0 * label - 1.0)


def binary_perceptron_loss(label: int, score: float) -> float:
  """Binary perceptron loss.

  Args:
    label: ground-truth integer label (0 or 1).
    score: score produced by the model (float).
  Returns:
    loss value.

  References:
    https://en.wikipedia.org/wiki/Perceptron
  """
  return optax_losses.perceptron_loss(score, 2.0 * label - 1.0)

# Multiclass classification.


def multiclass_logistic_loss(label: int, logits: jnp.ndarray) -> float:
  """Multiclass logistic loss.

  Args:
    label: ground-truth integer label, between 0 and n_classes - 1.
    logits: scores produced by the model, shape = (n_classes, ).
  Returns:
    loss value
  """
  return optax_losses.softmax_cross_entropy_with_integer_labels(
      jnp.asarray(logits), jnp.asarray(label))


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
  scores = jnp.asarray(scores)
  proba = projection_simplex(scores)
  # Fenchel conjugate of the Gini negentropy, defined by:
  # cumulant = jnp.dot(proba, scores) + 0.5 * jnp.dot(proba, (1 - proba)).
  scores = (scores - scores[label]).at[label].set(0.0)
  return (jnp.dot(proba, jnp.where(proba, scores, 0.0))
          + 0.5 * (1.0 - jnp.dot(proba, proba)))


def multiclass_hinge_loss(label: int,
                          scores: jnp.ndarray) -> float:
  """Multiclass hinge loss.

  Args:
    label: ground-truth integer label.
    scores: scores produced by the model (floats).
  Returns:
    loss value

  References:
    https://en.wikipedia.org/wiki/Hinge_loss
  """
  one_hot_label = jax.nn.one_hot(label, scores.shape[0])
  return jnp.max(scores + 1.0 - one_hot_label) - jnp.dot(scores, one_hot_label)


def multiclass_perceptron_loss(label: int,
                               scores: jnp.ndarray) -> float:
  """Binary perceptron loss.

  Args:
    label: ground-truth integer label.
    scores: score produced by the model (float).
  Returns:
    loss value.

  References:
    Michael Collins. Discriminative training methods for Hidden Markov Models:
    Theory and experiments with perceptron algorithms. EMNLP 2002
  """
  one_hot_label = jax.nn.one_hot(label, scores.shape[0])
  return jnp.max(scores) - jnp.dot(scores, one_hot_label)

# Fenchel-Young losses


def make_fenchel_young_loss(max_fun: Callable[[jnp.ndarray], float]):
  """Creates a Fenchel-Young loss from a max function.

  Args:
    max_fun: the max function on which the Fenchel-Young loss is built.

  Returns:
    A Fenchel-Young loss function with the same signature.

  Example:
    Given a max function, e.g. the log sum exp

      from jax.scipy.special import logsumexp

      FY_loss = make_fy_loss(max_fun=logsumexp)

    Then FY loss is the Fenchel-Young loss, given for F = max_fun by

      FY_loss(y_true, scores) = F(scores) - <scores, y_true>

    Its gradient, computed automatically, is given by

      grad FY_loss = y_eps(scores) - y_true

    where y_eps is the gradient of F, the argmax.
  """

  def fy_loss(y_true, scores, *args, **kwargs):
    return optax_losses.make_fenchel_young_loss(max_fun)(
        scores.ravel(), y_true.ravel(), *args, **kwargs)
  return fy_loss
