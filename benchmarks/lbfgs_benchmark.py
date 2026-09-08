# Copyright 2022 Google LLC
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

"""Benchmark LBFGS implementation."""


from absl import app
from absl import flags

from sklearn import datasets

import jax.numpy as jnp
import jaxopt

import numpy as onp

import matplotlib.pyplot as plt

from flax import linen as nn
import tensorflow_datasets as tfds
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_integer("maxiter", default=30, help="Max # of iterations.")
flags.DEFINE_integer("n_samples", default=10000, help="Number of samples.")
flags.DEFINE_integer("n_features", default=200, help="Number of features.")
flags.DEFINE_string("task", "binary_logreg", "Task to benchmark.")
flags.DEFINE_integer("batch_size", default=1024, help="Batch size.")
flags.DEFINE_string("dataset", default="mnist", help="Dataset to use.")


def binary_logreg(linesearch):
  X, y = datasets.make_classification(n_samples=FLAGS.n_samples,
                                      n_features=FLAGS.n_features,
                                      n_classes=2,
                                      n_informative=3,
                                      random_state=0)
  data = (X, y)
  fun = jaxopt.objective.binary_logreg
  init = jnp.zeros(X.shape[1])
  lbfgs = jaxopt.LBFGS(fun=fun, linesearch=linesearch)
  state = lbfgs.init_state(init, data=data)
  errors = onp.zeros(FLAGS.maxiter)
  params = init

  for it in range(FLAGS.maxiter):
    params, state = lbfgs.update(params, state, data=data)
    errors[it] = state.error

  return errors


def multiclass_logreg(linesearch):
  X, y = datasets.make_classification(n_samples=FLAGS.n_samples,
                                      n_features=FLAGS.n_features,
                                      n_classes=5,
                                      n_informative=5,
                                      random_state=0)
  data = (X, y)
  fun = jaxopt.objective.multiclass_logreg
  init = jnp.zeros((X.shape[1], 5))
  lbfgs = jaxopt.LBFGS(fun=fun, linesearch=linesearch)
  state = lbfgs.init_state(init, data=data)
  errors = onp.zeros(FLAGS.maxiter)
  params = init

  for it in range(FLAGS.maxiter):
    params, state = lbfgs.update(params, state, data=data)
    errors[it] = state.error

  return errors


def run_binary_logreg():
  errors_backtracking = binary_logreg("backtracking")
  errors_zoom = binary_logreg("zoom")

  plt.figure()
  plt.plot(jnp.arange(FLAGS.maxiter), errors_backtracking, label="backtracking")
  plt.plot(jnp.arange(FLAGS.maxiter), errors_zoom, label="zoom")
  plt.xlabel("Iterations")
  plt.ylabel("Gradient error")
  plt.yscale("log")
  plt.legend(loc="best")
  plt.show()


def run_multiclass_logreg():
  errors_backtracking = multiclass_logreg("backtracking")
  errors_zoom = multiclass_logreg("zoom")

  plt.figure()
  plt.plot(jnp.arange(FLAGS.maxiter), errors_backtracking, label="backtracking")
  plt.plot(jnp.arange(FLAGS.maxiter), errors_zoom, label="zoom")
  plt.xlabel("Iterations")
  plt.ylabel("Gradient error")
  plt.yscale("log")
  plt.legend(loc="best")
  plt.show()


def cnn(linesearch):

  def load_dataset(dataset, batch_size):
    """Loads the dataset as a generator of batches."""
    train_ds, ds_info = tfds.load(f"{dataset}:3.*.*", split="train",
                                  as_supervised=True, with_info=True)
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(10 * batch_size, seed=0)
    train_ds = train_ds.batch(batch_size)
    return tfds.as_numpy(train_ds), ds_info

  class CNN(nn.Module):
    """A simple CNN model."""
    num_classes: int
    net_width: int

    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=self.net_width, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = nn.Conv(features=self.net_width*2, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = x.reshape((x.shape[0], -1))  # flatten
      x = nn.Dense(features=self.net_width*4)(x)
      x = nn.relu(x)
      x = nn.Dense(features=self.num_classes)(x)
      return x

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  train_ds, ds_info = load_dataset(FLAGS.dataset, FLAGS.batch_size)
  train_ds = iter(train_ds)

  # Initialize parameters.
  input_shape = (1,) + ds_info.features["image"].shape
  rng = jax.random.PRNGKey(0)
  num_classes = ds_info.features["label"].num_classes
  net_width = 4
  net = CNN(num_classes, net_width)

  logistic_loss = jax.vmap(jaxopt.loss.multiclass_logistic_loss)

  def loss_fun(params, data):
    """Compute the loss of the network."""
    inputs, labels = data
    x = inputs.astype(jnp.float32) / 255.
    logits = net.apply({"params": params}, x)
    loss_value = jnp.mean(logistic_loss(labels, logits))
    return loss_value

  net = CNN(num_classes, 4)
  params = net.init(rng, jnp.zeros(input_shape))["params"]

  opt = jaxopt.LBFGS(fun=loss_fun, linesearch=linesearch)
  state = opt.init_state(params)
  jitted_update = jax.jit(opt.update)

  errors = onp.zeros(FLAGS.maxiter)

  for it in range(FLAGS.maxiter):
    batch = next(train_ds)
    params, state = jitted_update(params, state, batch)
    errors[it] = state.error

  return errors


def run_cnn():
  errors_backtracking = cnn("backtracking")
  errors_zoom = cnn("zoom")

  plt.figure()
  plt.plot(jnp.arange(FLAGS.maxiter), errors_backtracking, label="backtracking")
  plt.plot(jnp.arange(FLAGS.maxiter), errors_zoom, label="zoom")
  plt.xlabel("Iterations")
  plt.ylabel("Gradient error")
  plt.yscale("log")
  plt.legend(loc="best")
  plt.show()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print("n_samples:", FLAGS.n_samples)
  print("n_features:", FLAGS.n_features)
  print("maxiter:", FLAGS.maxiter)
  print("task:", FLAGS.task)
  print()

  if FLAGS.task == "binary_logreg":
    run_binary_logreg()

  elif FLAGS.task == "multiclass_logreg":
    run_multiclass_logreg()

  elif FLAGS.task == "cnn":
    run_cnn()

  else:
    raise ValueError("Invalid task name.")


if __name__ == '__main__':
  app.run(main)
