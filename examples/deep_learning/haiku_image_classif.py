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

"""
Image classification example with Haiku and JAXopt.
===================================================
"""
import functools

from absl import app
from absl import flags

import haiku as hk

import jax
import jax.numpy as jnp

from jaxopt import loss
from jaxopt import ArmijoSGD
from jaxopt import OptaxSolver
from jaxopt import PolyakSGD
from jaxopt import tree_util

import optax

import tensorflow_datasets as tfds


dataset_names = [
    "mnist", "kmnist", "emnist", "fashion_mnist", "cifar10", "cifar100"
]


flags.DEFINE_float("l2reg", 1e-4, "L2 regularization.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate (used in adam).")
flags.DEFINE_bool("manual_loop", False, "Whether to use a manual training loop.")
flags.DEFINE_integer("maxiter", 100, "Maximum number of iterations.")
flags.DEFINE_float("max_stepsize", 0.1, "Maximum step size (used in polyak-sgd, armijo-sgd).")
flags.DEFINE_float("aggressiveness", 0.5, "Aggressiveness of line search in armijo-sgd.")
flags.DEFINE_float("momentum", 0.9, "Momentum strength (used in adam, polyak-sgd, armijo-sgd).")
flags.DEFINE_enum("dataset", "mnist", dataset_names, "Dataset to train on.")
flags.DEFINE_enum("model", "cnn", ["cnn", "mlp"], "Model architecture.")
flags.DEFINE_enum("solver", "adam", ["adam", "sgd", "polyak-sgd", "armijo-sgd"], "Solver to use.")
FLAGS = flags.FLAGS


def load_dataset(split, *, is_training, batch_size):
  """Loads the dataset as a generator of batches."""
  version = 3
  ds, ds_info = tfds.load(
      f"{FLAGS.dataset}:{version}.*.*",
      as_supervised=True,  # remove useless keys
      split=split,
      with_info=True)
  ds = ds.cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds)), ds_info


def net_fun(batch, num_classes):
  """Create model."""
  x = batch[0].astype(jnp.float32) / 255.
  if FLAGS.model == "cnn":
    model = hk.Sequential([
        hk.Conv2D(output_channels=32, kernel_shape=(3, 3), padding="SAME"),
        jax.nn.relu,
        hk.AvgPool(window_shape=(2, 2), strides=(2, 2), padding="SAME"),
        hk.Conv2D(output_channels=64, kernel_shape=(3, 3), padding="SAME"),
        jax.nn.relu,
        hk.AvgPool(window_shape=(2, 2), strides=(2, 2), padding="SAME"),
        hk.Flatten(),
        hk.Linear(256),
        jax.nn.relu,
        hk.Linear(num_classes),
    ])
  else:
    model = hk.Sequential([
        hk.Flatten(),
        hk.Linear(300),
        jax.nn.relu,
        hk.Linear(100),
        jax.nn.relu,
        hk.Linear(num_classes),
    ])
  y = model(x)
  return y


def main(argv):
  del argv

  train_ds, ds_info = load_dataset("train", is_training=True, batch_size=256)
  test_ds, _ = load_dataset("test", is_training=False, batch_size=1024)
  num_classes = ds_info.features["label"].num_classes

  # Initialize parameters.
  net = functools.partial(net_fun, num_classes=num_classes)
  net = hk.without_apply_rng(hk.transform(net))

  logistic_loss = jax.vmap(loss.multiclass_logistic_loss)

  def loss_fun(params, l2reg, data):
    """Compute the loss of the network."""
    logits = net.apply(params, data)
    _, labels = data
    sqnorm = tree_util.tree_l2_norm(params, squared=True)
    loss_value = jnp.mean(logistic_loss(labels, logits))
    return loss_value + 0.5 * l2reg * sqnorm

  @jax.jit
  def accuracy(params, data):
    _, labels = data
    predictions = net.apply(params, data)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == labels)

  def print_accuracy(params, state, *args, **kwargs):
    if state.iter_num % 10 == 0:
      # Periodically evaluate classification accuracy on test set.
      test_accuracy = accuracy(params, next(test_ds))
      test_accuracy = jax.device_get(test_accuracy)
      print(f"[Step {state.iter_num}] Test accuracy: {test_accuracy:.3f}.")
    return params, state

  # Initialize solver.

  if FLAGS.solver == "adam":
    # Equivalent to:
    # opt = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
    #                   optax.scale(-FLAGS.learning_rate))
    opt = optax.adam(FLAGS.learning_rate)
    solver = OptaxSolver(opt=opt, fun=loss_fun, maxiter=FLAGS.maxiter,
                         pre_update=print_accuracy)

  elif FLAGS.solver == "sgd":
    opt = optax.sgd(FLAGS.learning_rate, FLAGS.momentum)
    solver = OptaxSolver(opt=opt, fun=loss_fun, maxiter=FLAGS.maxiter,
                         pre_update=print_accuracy)


  elif FLAGS.solver == "polyak-sgd":
    solver = PolyakSGD(fun=loss_fun, maxiter=FLAGS.maxiter,
                       momentum=FLAGS.momentum,
                       max_stepsize=FLAGS.max_stepsize,
                       pre_update=print_accuracy)


  elif FLAGS.solver == "armijo-sgd":
    solver = ArmijoSGD(fun=loss_fun, maxiter=FLAGS.maxiter,
                       aggressiveness=FLAGS.aggressiveness,
                       momentum=FLAGS.momentum,
                       max_stepsize=FLAGS.max_stepsize,
                       pre_update=print_accuracy)

  else:
    raise ValueError("Unknown solver: %s" % FLAGS.solver)

  params = net.init(jax.random.PRNGKey(42), next(train_ds))

  # Run training loop.

  # In JAXopt, stochastic solvers can be run either using a manual for loop or
  # using `run_iterator`. We include both here for demonstration purpose.
  if FLAGS.manual_loop:
    state = solver.init_state(params)

    for _ in range(FLAGS.maxiter):
      params, state = solver.update(params=params, state=state,
                                    l2reg=FLAGS.l2reg,
                                    data=next(train_ds))

  else:
    params, state = solver.run_iterator(
        init_params=params, iterator=train_ds, l2reg=FLAGS.l2reg)

  print_accuracy(params=params, state=state)


if __name__ == "__main__":
  app.run(main)
