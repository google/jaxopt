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
MNIST example with Flax and JAXopt.
===================================
"""

from absl import app
from absl import flags

from flax import linen as nn

import jax
import jax.numpy as jnp

from jaxopt import loss
from jaxopt import OptaxSolver
from jaxopt import PolyakSGD
from jaxopt import tree_util

import optax

import tensorflow_datasets as tfds


flags.DEFINE_float("l2reg", 1e-4, "L2 regularization.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate (used in adam).")
flags.DEFINE_bool("manual_loop", False, "Whether to use a manual training loop.")
flags.DEFINE_integer("maxiter", 100, "Maximum number of iterations.")
flags.DEFINE_float("max_step_size", 0.1, "Maximum step size (used in polyak-sgd).")
flags.DEFINE_float("momentum", 0.9, "Momentum strength (used in adam, polyak-sgd).")
flags.DEFINE_enum("solver", "adam", ["adam", "sgd", "polyak-sgd"], "Solver to use.")
FLAGS = flags.FLAGS


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
def accuracy(params, data):
  x = data["image"].astype(jnp.float32) / 255.
  logits = net.apply({"params": params}, x)
  return jnp.mean(jnp.argmax(logits, axis=-1) == data["label"])


logistic_loss = jax.vmap(loss.multiclass_logistic_loss)


def loss_fun(params, l2reg, data):
  """Compute the loss of the network."""
  x = data["image"].astype(jnp.float32) / 255.
  logits = net.apply({"params": params}, x)
  labels = data["label"]
  sqnorm = tree_util.tree_l2_norm(params, squared=True)
  loss_value = jnp.mean(logistic_loss(labels, logits))
  return loss_value + 0.5 * l2reg * sqnorm


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
  if FLAGS.solver == "adam":
    solver = OptaxSolver(opt=optax.adam(1e-3), fun=loss_fun,
                         maxiter=FLAGS.maxiter, pre_update=pre_update)

  elif FLAGS.solver == "sgd":
    opt = optax.sgd(FLAGS.learning_rate, FLAGS.momentum)
    solver = OptaxSolver(opt=opt, fun=loss_fun,
                         maxiter=FLAGS.maxiter, pre_update=pre_update)


  elif FLAGS.solver == "polyak-sgd":
    solver = PolyakSGD(fun=loss_fun, maxiter=FLAGS.maxiter,
                       momentum=FLAGS.momentum,
                       max_step_size=FLAGS.max_step_size,
                       pre_update=pre_update)

  else:
    raise ValueError("Unknown solver: %s" % FLAGS.solver)

  rng = jax.random.PRNGKey(0)
  init_params = CNN().init(rng, jnp.ones([1, 28, 28, 1]))["params"]

  # Run training loop.

  # In JAXopt, stochastic solvers can be run either using a manual for loop or
  # using `run_iterator`. We include both here for demonstration purpose.
  if FLAGS.manual_loop:
    params, state = solver.init(init_params)

    for _ in range(FLAGS.maxiter):
      params, state = pre_update(params=params, state=state)
      params, state = solver.update(params=params, state=state,
                                    l2reg=FLAGS.l2reg,
                                    data=next(train_ds))

  else:
    solver.run_iterator(init_params=init_params,
                        iterator=train_ds,
                        l2reg=FLAGS.l2reg)

if __name__ == "__main__":
  app.run(main)
