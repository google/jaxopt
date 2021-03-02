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

"""Proximity operators."""

import jax
import jax.numpy as jnp


def prox_l1(x, alpha=1.0):
  """Proximal operator for the l1 norm, i.e., soft-thresholding operator.

  Args:
    x: input, shape = (size,).
    alpha: regularization strength, float or array of shape (size,).
  Returns:
    output, shape = (size,).
  """
  return jnp.sign(x) * jnp.maximum(jnp.abs(x) - alpha, 0)
