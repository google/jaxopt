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
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from jaxopt.projection import projection_simplex


def multiclass_logistic_loss(label: int, logits: jnp.ndarray) -> float:
  """Multiclass logistic loss.

  Args:
    label: ground-truth integer label, between 0 and n_classes - 1.
    logits: scores produced by the model, shape = (n_classes, ).
  Returns:
    loss value (float)
  """
  n_classes = logits.shape[0]
  one_hot = jax.nn.one_hot(label, n_classes)
  return logsumexp(logits) - jnp.dot(logits, one_hot)


def multiclass_sparsemax_loss(label: int, logits: jnp.ndarray) -> float:
  """Multiclass sparsemax loss.

  Args:
    label: ground-truth integer label, between 0 and n_classes - 1.
    logits: scores produced by the model, shape = (n_classes, ).
  Returns:
    loss value (float)
  """
  n_classes = logits.shape[0]
  one_hot = jax.nn.one_hot(label, n_classes)
  proba = projection_simplex(logits)
  # Fenchel conjugate of the Gini negentropy, defined by:
  # cumulant = jnp.dot(proba, logits) + 0.5 * jnp.dot(proba, (1 - proba)).
  cumulant = 0.5 + jnp.dot(proba , logits - 0.5 * proba)
  return cumulant - jnp.dot(logits, one_hot)
