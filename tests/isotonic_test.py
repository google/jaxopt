# Copyright 2023 Google LLC
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

"""Tests for Isotonic Regression."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from jax.test_util import check_grads
from jaxopt.isotonic import isotonic_l2_pav
from jaxopt._src import test_util
from sklearn import isotonic


class IsotonicPavTest(test_util.JaxoptTestCase):
  """Tests for PAV in JAX."""

  def test_output_shape_and_dtype(self, n=10):
    """Verifies the shapes and dtypes of output."""
    y = jax.random.normal(jax.random.PRNGKey(0), (n,))
    output = isotonic_l2_pav(y)
    self.assertEqual(output.shape, y.shape)
    self.assertEqual(output.dtype, y.dtype)

  @parameterized.product(increasing=[True, False])
  def test_compare_with_sklearn(self, increasing,  n=10):
    """Compares the output with the one of sklearn."""
    y = jax.random.normal(jax.random.PRNGKey(0), (n,))
    output = isotonic_l2_pav(y, increasing=increasing)
    output_sklearn = jnp.array(isotonic.isotonic_regression(y, increasing=increasing))
    self.assertArraysAllClose(output, output_sklearn)
    y_sort = y.sort()
    y_min = y_sort[2]
    y_max = y_sort[n-5]
    output = isotonic_l2_pav(y, y_min=y_min, y_max=y_max, increasing=increasing)
    output_sklearn = jnp.array(isotonic.isotonic_regression(y, y_min=y_min,
     y_max=y_max, increasing=increasing))
    self.assertArraysAllClose(output, output_sklearn)

  @parameterized.product(increasing=[True, False])
  def test_gradient(self, increasing, n=10):
    """Checks the gradient with finite differences."""
    y = jax.random.normal(jax.random.PRNGKey(0), (n,))

    def loss(y):
      return (isotonic_l2_pav(y**3, increasing=increasing)
       + isotonic_l2_pav(y, increasing=increasing) ** 2).mean()

    check_grads(loss, (y,), order=2)

  def test_gradient_min_max(self, n=10):
    """Checks the gradient with finite differences."""
    y = jax.random.normal(jax.random.PRNGKey(0), (n,))
    y_sort = y.sort()
    y_min = y_sort[2]
    y_max = y_sort[n-5]
    def loss(y):
      return (isotonic_l2_pav(y**3, y_min=y_min, y_max=y_max)
       + isotonic_l2_pav(y, y_min=y_min, y_max=y_max) ** 2).mean()

    check_grads(loss, (y,), order=2)

  def test_vmap(self, n_features=10, n_batches=16):
    """Verifies vmap."""
    y = jax.random.normal(jax.random.PRNGKey(0), (n_batches, n_features))
    isotonic_l2_pav_vmap = jax.vmap(isotonic_l2_pav)
    for i in range(n_batches):
      self.assertArraysAllClose(isotonic_l2_pav_vmap(y)[i], isotonic_l2_pav(y[i]))

if __name__ == '__main__':
  absltest.main()
