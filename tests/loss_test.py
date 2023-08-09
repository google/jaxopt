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
import jax.numpy as jnp
from jax.scipy.special import expit as sigmoid
from jax.scipy.special import logsumexp
from jaxopt import loss
from jaxopt import projection
from jaxopt._src import test_util
import numpy as onp


def one_hot_argmax(inputs: jnp.ndarray) -> jnp.ndarray:
  """An argmax one-hot function for arbitrary shapes."""
  inputs_flat = jnp.reshape(inputs, (-1))
  flat_one_hot = jax.nn.one_hot(jnp.argmax(inputs_flat), inputs_flat.shape[0])
  return jnp.reshape(flat_one_hot, inputs.shape)


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

  def test_binary_sparsemax_loss(self):
    def reference_impl(label: int, logit: float) -> float:
      scores = -(2*label-1)*logit
      if scores <= -1.0:
        return 0.0
      elif scores >= 1.0:
        return scores
      else:
        return (scores + 1.0) ** 2 / 4

    self._test_binary_loss_function(
        loss.binary_sparsemax_loss, loss.sparse_sigmoid, reference_impl
    )

  def test_binary_hinge_loss(self):
    def reference_impl(label: int, logit: float) -> float:
      return jax.nn.relu(1 - logit * (2.0 * label - 1.0))
    self._test_binary_loss_function(loss.binary_hinge_loss, jnp.sign,
                                    reference_impl)

  def test_perceptron_loss(self):
    def reference_impl(label: int, logit: float) -> float:
      return jax.nn.relu(- logit * (2.0 * label - 1.0))
    self._test_binary_loss_function(loss.binary_perceptron_loss, jnp.sign,
                                    reference_impl) 

  def _test_multiclass_loss_function(
      self, loss_fun, inv_link_fun, reference_impl, large_inputs_behavior=True,
      incorrect_label_infty=True
  ):
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
    if large_inputs_behavior:
      loss_val = loss_fun(0, jnp.array([1e9, 1e9]))
      expected = loss_fun(0, jnp.array([1, 1]))
      self.assertAllClose(loss_val, expected)

    # Check that -inf for incorrect label has no impact.
    if incorrect_label_infty:
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

  def test_multiclass_hinge_loss(self):
    def reference_impl(label, scores):
      one_hot_label = jax.nn.one_hot(label, scores.shape[-1])
      return jnp.max(scores + 1.0 - one_hot_label) - scores[label]
    def inv_link_fun(scores):
      return jax.nn.one_hot(jnp.argmax(scores), scores.shape[-1])

    self._test_multiclass_loss_function(loss.multiclass_hinge_loss,
                                        inv_link_fun,
                                        reference_impl,
                                        large_inputs_behavior=False,
                                        incorrect_label_infty=False)

  def test_multiclass_perceptron_loss(self):
    def reference_impl(label, scores):
      return jnp.max(scores) - scores[label]
    def inv_link_fun(scores):
      return jax.nn.one_hot(jnp.argmax(scores), scores.shape[-1])

    self._test_multiclass_loss_function(loss.multiclass_perceptron_loss,
                                        inv_link_fun,
                                        reference_impl,
                                        incorrect_label_infty=False)

  def test_huber(self):
    self.assertAllClose(0.0, loss.huber_loss(0, 0, .1))
    self.assertAllClose(0.0, loss.huber_loss(1, 1, .1))
    self.assertAllClose(.1 * (1.0- .5 * .1), loss.huber_loss(4, 3, .1))
    self.assertAllClose(0.125, loss.huber_loss(0, .5))

  def test_fenchel_young_reg(self):
    # Checks the behavior of the Fenchel-Young loss.
    fy_loss = loss.make_fenchel_young_loss(logsumexp)
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 2)
    theta_true = jax.random.uniform(rngs[0], (8, 5))
    y_true = jax.vmap(jax.nn.softmax)(theta_true)
    theta_random = jax.random.uniform(rngs[1], (8, 5))
    y_random = jax.vmap(jax.nn.softmax)(theta_random)
    fy_min = jax.vmap(fy_loss)(y_true, theta_true)
    fy_random = jax.vmap(fy_loss)(y_true, theta_random)
    # Checks that the loss is minimized for true value of the parameters.
    self.assertGreater(fy_random[0], fy_min[0])
    self.assertGreater(jnp.mean(fy_random), jnp.mean(fy_min))
    grad_random = jax.vmap(jax.grad(fy_loss, argnums=1))(y_true, theta_random)
    # Checks that the gradient of the loss takes the correct form.
    self.assertArraysAllClose(grad_random, y_random - y_true)
    y_one_hot = jax.vmap(one_hot_argmax)(theta_true)
    int_one_hot = jnp.where(y_one_hot == 1.)[1]
    loss_one_hot = jax.vmap(fy_loss)(y_one_hot, theta_random)
    log_loss = jax.vmap(loss.multiclass_logistic_loss)(int_one_hot,
                                                       theta_random)
    # Checks that the FY loss associated to logsumexp is correct.
    self.assertArraysAllClose(loss_one_hot, log_loss)


if __name__ == '__main__':
  absltest.main()
