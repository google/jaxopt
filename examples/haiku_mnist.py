# Copyright 2020 DeepMind Technologies Limited
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

"""MNIST example with Haiku and JAXopt."""

from absl import app
from absl import flags
import haiku as hk
import jax
import jax.numpy as jnp
from jaxopt import loss
from jaxopt import OptaxSolver
from jaxopt import tree_util
import optax
import tensorflow_datasets as tfds


flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
FLAGS = flags.FLAGS


def load_dataset(split, *, is_training, batch_size):
  """Loads the dataset as a generator of batches."""
  ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds))


def net_fun(batch):
  """Standard LeNet-300-100 MLP network."""
  x = batch["image"].astype(jnp.float32) / 255.
  mlp = hk.Sequential([
      hk.Flatten(),
      hk.Linear(300), jax.nn.relu,
      hk.Linear(100), jax.nn.relu,
      hk.Linear(10),
  ])
  return mlp(x)


net = hk.without_apply_rng(hk.transform(net_fun))


@jax.jit
def accuracy(params, data):
  predictions = net.apply(params, data)
  return jnp.mean(jnp.argmax(predictions, axis=-1) == data["label"])

logistic_loss = jax.vmap(loss.multiclass_logistic_loss)


def loss_fun(params, l2_regul, data):
  """Compute the loss of the network."""
  logits = net.apply(params, data)
  labels = data["label"]
  sqnorm = tree_util.tree_l2_norm(params, squared=True)
  loss_value = jnp.mean(logistic_loss(labels, logits))
  return loss_value + 0.5 * l2_regul * sqnorm


def main(argv):
  del argv

  train_ds = load_dataset("train", is_training=True, batch_size=1000)
  test_ds = load_dataset("test", is_training=False, batch_size=10000)

  def pre_update(params, state, *args, **kwargs):
    if state.iter_num % 10 == 0:
      # Periodically evaluate classification accuracy on test set.
      test_accuracy = accuracy(params, next(test_ds))
      test_accuracy = jax.device_get(test_accuracy)
      print(f"[Step {state.iter_num}] Test accuracy: {test_accuracy:.3f}.")
    return params, state

  # Initialize solver and parameters.

  # Equilent to:
  # opt = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
  #                   optax.scale(-FLAGS.learning_rate))
  opt = optax.adam(FLAGS.learning_rate)
  solver = OptaxSolver(opt=opt, fun=loss_fun, maxiter=100,
                       pre_update_fun=pre_update)
  init_params = net.init(jax.random.PRNGKey(42), next(train_ds))
  l2_regul = 1e-4

  # Run training loop.

  # Equivalent to:
  # params, state = opt.init(init_params)
  # for _ in range(100):
  #   params, state = opt.update(params=params, state=state,
  #                              l2_regul=l2_regul, data=next(train_ds))
  # except that implicit diff w.r.t. `l2_regul` will be supported.
  solver.run_iterator(init_params=init_params,
                      iterator=train_ds,
                      l2_regul=l2_regul)


if __name__ == "__main__":
  app.run(main)
