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

"""MNIST example with Flax and JAXopt."""

from absl import app
from flax import linen as nn
import jax
import jax.numpy as jnp
from jaxopt import loss
from jaxopt import optax_wrapper
from jaxopt import tree_util
import optax
import tensorflow_datasets as tfds


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


def main(argv):
  net = CNN()

  @jax.jit
  def accuracy(params, batch):
    x = batch["image"].astype(jnp.float32) / 255.
    logits = net.apply({"params": params}, x)
    return jnp.mean(jnp.argmax(logits, axis=-1) == batch["label"])

  logistic_loss = jax.vmap(loss.multiclass_logistic_loss)

  # Objective / loss functions in JAXopt must have the form
  # loss_fun(params, hyperparams, data).
  def loss_fun(params, l2_regul, batch):
    """Compute the loss of the network."""
    x = batch["image"].astype(jnp.float32) / 255.
    logits = net.apply({"params": params}, x)
    labels = batch["label"]
    sqnorm = tree_util.tree_l2_norm(params, squared=True)
    loss_value = jnp.mean(logistic_loss(labels, logits))
    return loss_value + 0.5 * l2_regul * sqnorm

  train_ds = load_dataset("train", is_training=True, batch_size=1000)
  test_ds = load_dataset("test", is_training=False, batch_size=10000)

  def pre_update(params, state, hyperparams, data):
    if state.iter_num % 10 == 0:
      # Periodically evaluate classification accuracy on test set.
      test_accuracy = accuracy(params, next(test_ds))
      test_accuracy = jax.device_get(test_accuracy)
      print(f"[Step {state.iter_num}] Test accuracy: {test_accuracy:.3f}.")
    return params, state

  # Initialize solver and parameters.
  solver = optax_wrapper.OptaxSolver(opt=optax.adam(1e-3),
                                     fun=loss_fun,
                                     maxiter=100,
                                     pre_update_fun=pre_update)
  rng = jax.random.PRNGKey(0)
  init_params = CNN().init(rng, jnp.ones([1, 28, 28, 1]))["params"]
  l2_regul = 1e-4

  # Run training loop.

  # Equivalent to:
  # params, state = opt.init(init_params)
  # for _ in range(100):
  #   params, state = opt.update(params=params, state=state,
  #                              hyperparams=l2_regul, data=next(train_ds))
  # except that implicit diff w.r.t. `hyperparams` will be supported.
  solver.run_iterator(hyperparams=l2_regul,
                      iterator=train_ds,
                      init_params=init_params)


if __name__ == "__main__":
  app.run(main)
