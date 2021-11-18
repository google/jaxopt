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

"""
Robust training.
================

The following code trains a convolutional neural network (CNN) to be robust
with respect to the fast sign gradient (FGSM) method.

The Fast Gradient Sign Method (FGSM) is a simple yet effective method to
generate adversarial images. It constructs an adversarial by adding a small
perturbation in the direction of the sign of the gradient with respect to the
input. The gradient ensures this perturbation locally maximizes the objective,
while the sign ensures that the update is on the boundary of the L-infinity
ball.

References
----------
  Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining
  and harnessing adversarial examples." https://arxiv.org/abs/1412.6572
"""

import tensorflow_datasets as tfds

import jax
from jax import numpy as jnp

from flax import linen as nn
import optax

from tqdm import tqdm

from jaxopt import loss
from jaxopt import OptaxSolver
from jaxopt import tree_util


def normalize(images):
  """Normalizes images to lie in (0,1) and be float32."""
  return jnp.asarray(images).astype(jnp.float32) / 255.

def load_datasets():
  """Load MNIST train and test datasets into memory.
  
  Taken from https://github.com/google/flax/blob/main/examples/mnist/train.py.
  """
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'], test_ds['image'] = map(normalize, (train_ds['image'], test_ds['image']))
  return train_ds, test_ds


def shuffle(ds, rng, batch_size):
  """Shuffles training data set, and returns batched examples."""
  n_train = len(ds['image'])
  rng_perm, rng = jax.random.split(rng)

  steps_per_epoch = n_train // batch_size

  permutation = jax.random.permutation(rng_perm, n_train)
  permutation = permutation[:steps_per_epoch * batch_size]  # skip incomplete batch
  permutation = permutation.reshape((steps_per_epoch, batch_size))

  images, labels = ds['image'][permutation, ...], ds['label'][permutation, ...]
  return zip(images, labels)

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
    x = jnp.reshape(x, (x.shape[0], -1))
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


@jax.jit
def loss_fun(params, l2_regul, images, labels):
  """Compute the loss of the network."""
  logits = net.apply({"params": params}, images)
  sqnorm = tree_util.tree_l2_norm(params, squared=True)
  loss_value = jnp.mean(logistic_loss(labels, logits))
  return loss_value + 0.5 * l2_regul * sqnorm


@jax.jit
def fsgm_attack(image, label, params, l2_regul, epsilon=0.1):
  """Fast sign-gradient attack on the L-infinity ball with radius epsilon.

  Parameters:
    image: array-like, input data for the CNN
    label: integer, class label corresponding to image
    params: tree, parameters of the model to attack
    l2_regul: float, L2 regularization scale in the loss function
    epsilon: float, radius of the L-infinity ball.

  Returns:
    perturbed_image: Adversarial image on the boundary of the L-infinity ball of radius
      epsilon and centered at image.
  """
  # compute gradient of the loss wrt to the image
  grad = jax.grad(loss_fun, argnums=2)(params, l2_regul, image, label)
  adv_image = image + epsilon * jnp.sign(grad)
  # clip the image to ensure pixels are between 0 and 1
  return jnp.clip(adv_image, 0, 1)

train_ds, test_ds = load_datasets()

# Initialize solver and parameters.
solver = OptaxSolver(opt=optax.adam(1e-3), fun=loss_fun)
rng = jax.random.PRNGKey(0)
params = CNN().init(rng, jnp.ones([1, 28, 28, 1]))["params"]
l2_regul = 1e-4

n_epochs = 10
batch_size = 128

state = solver.init_state(params)
rng = jax.random.PRNGKey(0)

for epoch in range(n_epochs):
  # Training
  pbar = tqdm(shuffle(train_ds, rng, batch_size=128),
              total=train_ds['label'].size // 128)
  acc = []
  adv_acc = []
  for images, labels in pbar:

    images_adv = fsgm_attack(images, labels, params, l2_regul=0.)

    # train on adversarial images
    params, state = solver.update(params=params, state=state,
                              l2_regul=l2_regul, images=images_adv, labels=labels)
    acc.append(accuracy(params, images, labels))
    adv_acc.append(accuracy(params, images_adv, labels))

  print(f"Train accuracy: {jnp.mean(jnp.array(acc)): .3f}")
  print(f"Train adversarial accuracy: {jnp.mean(jnp.array(adv_acc)): .3f}")

  # Testing
  images_test = test_ds['image']
  labels_test = test_ds['label']

  test_accuracy = accuracy(params, images_test, labels_test)
  print("Test accuracy:", test_accuracy)

  images_adv_test = fsgm_attack(images_test, labels_test, params, l2_regul=0.)
  test_adversarial_accuracy = accuracy(params, images_adv_test, labels_test)
  print("Test adversarial accuracy:", test_adversarial_accuracy)
  print()
