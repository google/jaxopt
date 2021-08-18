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

"""Robust Training in JAXOpt.

The following code trains a convolutional neural network (CNN) to be robust
with respect to the fast sign gradient (FGSM) method.

The Fast Gradient Sign Method (FGSM) is a simple yet effective method to generate
adversarial images. It constructs an adversarial by adding a small perturbation in
the direction of the sign of the gradient with respect to the input. The gradient
ensures this perturbation locally maximizes the objective, while the sign ensures
that the update is on the boundary of the L-infinity ball.


References:
  Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining
  and harnessing adversarial examples." https://arxiv.org/abs/1412.6572
"""

import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

import jax
from jax import numpy as jnp

from flax import linen as nn
import optax

from jaxopt import GradientDescent
from jaxopt import loss
from jaxopt import OptaxSolver
from jaxopt import tree_util


def load_dataset(split, *, is_training, batch_size):
  """Loads the dataset as a generator of batches."""
  ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds))


class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


net = CNN()


@jax.jit
def accuracy(params, images, labels):
  logits = net.apply({"params": params}, images)
  return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


logistic_loss = jax.vmap(loss.multiclass_logistic_loss)


def loss_fun(params, l2_regul, images, labels):
  """Compute the loss of the network."""
  logits = net.apply({"params": params}, images)
  sqnorm = tree_util.tree_l2_norm(params, squared=True)
  loss_value = jnp.mean(logistic_loss(labels, logits))
  return loss_value + 0.5 * l2_regul * sqnorm

train_ds = load_dataset("train", is_training=True, batch_size=128)
test_ds = load_dataset("test", is_training=False, batch_size=1000)

# Initialize solver and parameters.
solver = OptaxSolver(opt=optax.adam(1e-3), fun=loss_fun)
rng = jax.random.PRNGKey(0)
init_params = CNN().init(rng, jnp.ones([1, 28, 28, 1]))["params"]
l2_regul = 1e-4

params, state = solver.init(init_params)
for it in range(200):
  data = next(train_ds)
  images = data['image'].astype(jnp.float32) / 255
  labels = data['label']

  def fsgm_attack(image, label, epsilon=0.1):
    """Fast sign-gradient attack on the L-infinity ball with radius epsilon.
    
    Parameters:
      image: array-like, input data for the CNN
      label: integer, class label corresponding to image
      epsilon: float, radius of the L-infinity ball. 

    Returns:
      perturbed_image: Adversarial image on the boundary of the L-infinity ball of radius
        epsilon and centered at image.
    """
    # comppute gradient of the loss wrt to the image
    grad = jax.grad(loss_fun, argnums=2)(params, l2_regul, image, label)
    adv_image = image + epsilon * jnp.sign(grad)
    # clip the image to ensure pixels are between 0 and 1
    return jnp.clip(adv_image, 0, 1)

  images_adv = fsgm_attack(images, labels)

  # run adversarial training
  params, state = solver.update(params=params, state=state,
                             l2_regul=l2_regul, images=images_adv, labels=labels)
  
  if it % 10 == 0:
    data_test = next(test_ds)
    images_test = data_test['image'].astype(jnp.float32) / 255
    labels_test = data_test['label']

    test_accuracy = accuracy(params, images_test, labels_test)
    print("Accuracy on test set", test_accuracy)

    images_adv_test = fsgm_attack(images_test, labels_test)
    test_adversarial_accuracy = accuracy(params, images_adv_test, labels_test)
    print("Accuracy on adversarial images", test_adversarial_accuracy)
    print()
