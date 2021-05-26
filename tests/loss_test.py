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
from jax import test_util as jtu
from jax.nn import softmax
from jax.scipy.special import expit as sigmoid
import jax.numpy as jnp

from jaxopt.loss import binary_logistic_loss
from jaxopt.loss import huber_loss
from jaxopt.loss import multiclass_logistic_loss
from jaxopt.loss import multiclass_sparsemax_loss
from jaxopt.projection import projection_simplex


class LossTest(jtu.JaxTestCase):

  def _test_binary_loss_function(self, loss_fun, inv_link_fun):
    # Check that loss is zero when the weight goes to the correct label.
    loss = loss_fun(0, -1e5)
    self.assertEqual(loss, 0)
    loss = loss_fun(1, 1e5)
    self.assertEqual(loss, 0)

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

  def test_binary_logistic_loss(self):
    self._test_binary_loss_function(binary_logistic_loss, sigmoid)

  def _test_multiclass_loss_function(self, loss_fun, inv_link_fun):
    # Check that loss is zero when all weights goes to the correct label.
    loss = loss_fun(0, jnp.array([1e5, 0, 0]))
    self.assertEqual(loss, 0)

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

  def test_multiclass_logistic_loss(self):
    self._test_multiclass_loss_function(multiclass_logistic_loss, softmax)

  def test_multiclass_sparsemax_loss(self):
    self._test_multiclass_loss_function(multiclass_sparsemax_loss,
                                        projection_simplex)

  def test_huber(self):
    self.assertAllClose(0.0, huber_loss(0, 0, .1))
    self.assertAllClose(0.0, huber_loss(1, 1, .1))
    self.assertAllClose(.1 * (1.0- .5 * .1), huber_loss(4, 3, .1))
    self.assertAllClose(0.125, huber_loss(0, .5))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
