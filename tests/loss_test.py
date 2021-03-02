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
import jax.numpy as jnp
from jaxopt.loss import multiclass_logistic_loss
from jax.nn import softmax


class LossTest(jtu.JaxTestCase):

  def test_multiclass_logistic_loss(self):
    # Check that loss is zero when all weights goes to the correct label.
    loss = multiclass_logistic_loss(0, jnp.array([1e5, 0, 0]))
    self.assertEqual(loss, 0)

    # Check that gradient has the expected form.
    logits = jnp.array([1.2, 0.3, 2.3])
    grad = jax.grad(multiclass_logistic_loss, argnums=1)(0, logits)
    self.assertArraysAllClose(grad, softmax(logits) - jnp.array([1, 0, 0]))

    # Check that vmaps works as expected.
    logits = jnp.array([[1.3, 2.5],
                        [1.7, -0.3],
                        [0.2, 1.2]])
    labels = jnp.array([0, 0, 1])
    losses = jax.vmap(multiclass_logistic_loss)(labels, logits)
    losses2 = jnp.array([multiclass_logistic_loss(labels[0], logits[0]),
                         multiclass_logistic_loss(labels[1], logits[1]),
                         multiclass_logistic_loss(labels[2], logits[2])])
    self.assertArraysAllClose(losses, losses2)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
