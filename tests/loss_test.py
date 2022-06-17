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

from absl.testing import absltest

import jax
from jax.nn import softmax
from jax.nn import softplus
from jax.scipy.special import expit as sigmoid
from jax.scipy.special import logsumexp
import jax.numpy as jnp

from jaxopt import loss
from jaxopt import projection
from jaxopt._src import test_util

import numpy as onp


class LossTest(test_util.JaxoptTestCase):

  def _test_binary_loss_function(self, loss_fun, inv_link_fun, reference_impl):
    # Check that loss is zero when the weight goes to the correct label.
    loss_val = loss_fun(0, -1e5)
    self.assertEqual(loss_val, 0)
    loss_val = loss_fun(1, 1e5)
    self.assertEqual(loss_val, 0)

    # Check that gradient has the expected form.
    logit = 1.2
    grad = jax.grad(loss_fun, argnums=1)(0, logit)
    self.assertArraysAllClose(grad, inv_link_fun(logit))
    grad = jax.grad(loss_fun, argnums=1)(1, logit)
    self.assertArraysAllClose(grad, inv_link_fun(logit) - 1)

    # Check that vmap works as expected.
    logits = jnp.array([1.2, -0.3, 0.7])
    labels = jnp.array([0, 0, 1])
    losses = jax.vmap(loss_fun)(labels, logits)
    losses2 = jnp.array([loss_fun(labels[0], logits[0]),
                         loss_fun(labels[1], logits[1]),
                         loss_fun(labels[2], logits[2])])
    self.assertArraysAllClose(losses, losses2)

    # Compare against intuitive reference implementation.
    rng = onp.random.RandomState(0)
    for _ in range(10):
      logit = jnp.array(rng.randn(), dtype=jnp.float32)
      label = jnp.array(rng.randint(2))
      loss_val = loss_fun(label, logit)
      expected = reference_impl(label, logit)
      self.assertAllClose(loss_val, expected)

    # Check that correct value is obtained for large inputs.
    loss_val = loss_fun(0, -1e9)
    self.assertAllClose(loss_val, 0.0)
    loss_val = loss_fun(0, 1e9)
    self.assertAllClose(loss_val, 1e9)

    # Check that loss behaves correctly with infinite input.
    loss_val = loss_fun(1, jnp.inf)
    self.assertEqual(loss_val, 0.0)
    loss_val = loss_fun(0, -jnp.inf)
    self.assertEqual(loss_val, 0.0)
    loss_val = loss_fun(1, -jnp.inf)
    self.assertEqual(loss_val, jnp.inf)
    loss_val = loss_fun(0, jnp.inf)
    self.assertEqual(loss_val, jnp.inf)

    # Check that gradient has the expected form for large inputs.
    logit = 1e9
    grad = jax.grad(loss_fun, argnums=1)(0, logit)
    self.assertArraysAllClose(grad, inv_link_fun(logit))
    grad = jax.grad(loss_fun, argnums=1)(1, logit)
    self.assertArraysAllClose(grad, inv_link_fun(logit) - 1)

  def test_binary_logistic_loss(self):
    reference_impl = lambda label, logit: softplus(logit) - label * logit
    self._test_binary_loss_function(loss.binary_logistic_loss, sigmoid,
                                    reference_impl)

  def _test_multiclass_loss_function(self, loss_fun, inv_link_fun,
                                     reference_impl):
    # Check that loss is zero when all weights goes to the correct label.
    loss_val = loss_fun(0, jnp.array([1e5, 0, 0]))
    self.assertEqual(loss_val, 0)

    # Check that gradient has the expected form.
    logits = jnp.array([1.2, 0.3, 2.3])
    grad = jax.grad(loss_fun, argnums=1)(0, logits)
    self.assertArraysAllClose(grad, inv_link_fun(logits) - jnp.array([1, 0, 0]))

    # Check that vmap works as expected.
    logits = jnp.array([[1.3, 2.5],
                        [1.7, -0.3],
                        [0.2, 1.2]])
    labels = jnp.array([0, 0, 1])
    losses = jax.vmap(loss_fun)(labels, logits)
    losses2 = jnp.array([loss_fun(labels[0], logits[0]),
                         loss_fun(labels[1], logits[1]),
                         loss_fun(labels[2], logits[2])])
    self.assertArraysAllClose(losses, losses2)

    # Compare against intuitive reference implementation.
    rng = onp.random.RandomState(0)
    for _ in range(10):
      n_classes = rng.randint(8, 16, size=())
      logits = jnp.array(rng.randn(n_classes), dtype=jnp.float32)
      label = jnp.array(rng.randint(n_classes))
      loss_val = loss_fun(label, logits)
      expected = reference_impl(label, logits)
      self.assertAllClose(loss_val, expected)

    # Check that correct value is obtained for large inputs.
    loss_val = loss_fun(0, jnp.array([1e9, 1e9]))
    expected = loss_fun(0, jnp.array([1, 1]))
    self.assertAllClose(loss_val, expected)

    # Check that -inf for incorrect label has no impact.
    loss_val = loss_fun(0, jnp.array([0.0, 0.0, -jnp.inf]))
    expected = loss_fun(0, jnp.array([0.0, 0.0]))
    self.assertAllClose(loss_val, expected)
    # Check that -inf for correct label results in infinite loss.
    loss_val = loss_fun(0, jnp.array([-jnp.inf, 0.0, 0.0]))
    self.assertEqual(loss_val, jnp.inf)
    # Check that nan is contagious.
    loss_val = loss_fun(0, jnp.array([0.0, 0.0, jnp.nan]))
    self.assertTrue(jnp.isnan(loss_val))
    loss_val = loss_fun(0, jnp.array([jnp.nan, 0.0, 0.0]))
    self.assertTrue(jnp.isnan(loss_val))

    # Check that gradient has the expected form for large inputs.
    logits = 1e9 * jnp.array([1.2, 0.3, 2.3])
    grad = jax.grad(loss_fun, argnums=1)(0, logits)
    self.assertArraysAllClose(grad, inv_link_fun(logits) - jnp.array([1, 0, 0]))

    # Check that gradient has the expected form with -inf inputs.
    logits = jnp.array([1.2, -jnp.inf, 2.3])
    grad = jax.grad(loss_fun, argnums=1)(0, logits)
    self.assertArraysAllClose(grad, inv_link_fun(logits) - jnp.array([1, 0, 0]))

  def test_multiclass_logistic_loss(self):
    def reference_impl(label, logits):
      n_classes = logits.shape[0]
      one_hot = jax.nn.one_hot(label, n_classes)
      return logsumexp(logits) - jnp.dot(logits, one_hot)

    self._test_multiclass_loss_function(loss.multiclass_logistic_loss, softmax,
                                        reference_impl)

  def test_multiclass_sparsemax_loss(self):
    def reference_impl(label, scores):
      n_classes = scores.shape[0]
      one_hot = jax.nn.one_hot(label, n_classes)
      proba = projection.projection_simplex(scores)
      cumulant = 0.5 + jnp.dot(proba, scores - 0.5 * proba)
      return cumulant - jnp.dot(scores, one_hot)

    self._test_multiclass_loss_function(loss.multiclass_sparsemax_loss,
                                        projection.projection_simplex,
                                        reference_impl)

  def test_huber(self):
    self.assertAllClose(0.0, loss.huber_loss(0, 0, .1))
    self.assertAllClose(0.0, loss.huber_loss(1, 1, .1))
    self.assertAllClose(.1 * (1.0- .5 * .1), loss.huber_loss(4, 3, .1))
    self.assertAllClose(0.125, loss.huber_loss(0, .5))


if __name__ == '__main__':
  absltest.main()
