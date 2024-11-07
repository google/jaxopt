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
Deep Equilibrium (DEQ) model in Flax with Anderson acceleration.
================================================================

This implementation is strongly inspired by the Pytorch code snippets in [3].

A similar model called "implicit deep learning" is also proposed in [2].

In practice BatchNormalization and initialization of weights in convolutions are
important to ensure convergence.

[1] Bai, S., Kolter, J.Z. and Koltun, V., 2019. Deep Equilibrium Models.
Advances in Neural Information Processing Systems, 32, pp.690-701.

[2] El Ghaoui, L., Gu, F., Travacca, B., Askari, A. and Tsai, A., 2021.
Implicit deep learning. SIAM Journal on Mathematics of Data Science, 3(3),
pp.930-958.

[3] http://implicit-layers-tutorial.org/deep_equilibrium_models/
"""

from functools import partial
from typing import Any, Tuple, Callable

from absl import app
from absl import flags

from flax import linen as nn

import jax
import jax.numpy as jnp

from jaxopt import loss
from jaxopt import OptaxSolver
from jaxopt import FixedPointIteration
from jaxopt import AndersonAcceleration
from jaxopt.linear_solve import solve_gmres, solve_normal_cg
from jaxopt.tree_util import tree_add, tree_sub, tree_l2_norm

import optax

import tensorflow_datasets as tfds
import tensorflow as tf


dataset_names = [
    "mnist", "kmnist", "emnist", "fashion_mnist", "cifar10", "cifar100"
]

# training hyper-parameters
flags.DEFINE_float("l2reg", 0., "L2 regularization.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_integer("maxiter", 100, "Maximum number of iterations.")
flags.DEFINE_enum("dataset", "mnist", dataset_names, "Dataset to train on.")
flags.DEFINE_integer("net_width", 1, "Multiplicator of neural network width.")
flags.DEFINE_integer("evaluation_frequency", 1,
                     "Number of iterations between two evaluation measures.")
flags.DEFINE_integer("train_batch_size", 256, "Batch size at train time.")
flags.DEFINE_integer("test_batch_size", 1024, "Batch size at test time.")


solvers = ["normal_cg", "gmres", "anderson"]
flags.DEFINE_enum("backward_solver", "normal_cg", solvers,
                  "Solver of linear sytem in implicit differentiation.")

# anderson acceleration parameters
flags.DEFINE_enum("forward_solver", "anderson", ["anderson", "fixed_point"],
                  "Whether to use Anderson acceleration.")
flags.DEFINE_integer("forward_maxiter", 20, "Number of fixed point iterations.")
flags.DEFINE_float("forward_tol", 1e-2, "Tolerance in fixed point iterations.")
flags.DEFINE_integer("anderson_history_size", 5,
                     "Size of history in Anderson updates.")
flags.DEFINE_float("anderson_ridge", 1e-4,
                   "Ridge regularization in Anderson updates.")

FLAGS = flags.FLAGS


def load_dataset(split, *, is_training, batch_size):
  """Loads the dataset as a generator of batches."""
  ds, ds_info = tfds.load(f"{FLAGS.dataset}:3.*.*", split=split,
                          as_supervised=True, with_info=True)
  ds = ds.cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds)), ds_info


class ResNetBlock(nn.Module):
  """ResNet block."""

  channels: int
  channels_bottleneck: int
  num_groups: int = 8
  kernel_size: Tuple[int, int] = (3, 3)
  use_bias: bool = False
  act: Callable = nn.relu

  @nn.compact
  def __call__(self, z, x):
    # Note that stddev=0.01 is important to avoid divergence.
    # Empirically it ensures that fixed point iterations converge.
    y = z
    y = nn.Conv(features=self.channels_bottleneck, kernel_size=self.kernel_size,
                padding="SAME", use_bias=self.use_bias,
                kernel_init=jax.nn.initializers.normal(stddev=0.01))(y)
    y = self.act(y)
    y = nn.GroupNorm(num_groups=self.num_groups)(y)
    y = nn.Conv(features=self.channels, kernel_size=self.kernel_size,
                padding="SAME", use_bias=self.use_bias,
                kernel_init=jax.nn.initializers.normal(stddev=0.01))(y)
    y = y + x
    y = nn.GroupNorm(num_groups=self.num_groups)(y)
    y = y + z
    y = self.act(y)
    y = nn.GroupNorm(num_groups=self.num_groups)(y)
    return y


class DEQFixedPoint(nn.Module):
  """Batched computation of ``block`` using ``fixed_point_solver``."""

  block: Any  # nn.Module
  fixed_point_solver: Any  # AndersonAcceleration or FixedPointIteration

  @nn.compact
  def __call__(self, x):
    # shape of a single example
    init = lambda rng, x: self.block.init(rng, x[0], x[0])
    block_params = self.param("block_params", init, x)

    def block_apply(z, x, block_params):
      return self.block.apply(block_params, z, x)

    solver = self.fixed_point_solver(fixed_point_fun=block_apply)
    def batch_run(x, block_params):
      return solver.run(x, x, block_params)[0]

    # We use vmap since we want to compute the fixed point separately for each
    # example in the batch.
    return jax.vmap(batch_run, in_axes=(0,None), out_axes=0)(x, block_params)


class FullDEQ(nn.Module):
  """DEQ model."""

  num_classes: int
  channels: int
  channels_bottleneck: int
  fixed_point_solver: Callable

  @nn.compact
  def __call__(self, x, train):
    # Note that x is a batch of examples:
    # because of BatchNorm we cannot define the forward pass in the network for
    # a single example.
    x = nn.Conv(features=self.channels, kernel_size=(3,3), use_bias=True,
                padding="SAME")(x)
    x = nn.BatchNorm(use_running_average=not train, momentum=0.9,
                     epsilon=1e-5)(x)
    block = ResNetBlock(self.channels, self.channels_bottleneck)
    deq_fixed_point = DEQFixedPoint(block, self.fixed_point_solver)
    x = deq_fixed_point(x)
    x = nn.BatchNorm(use_running_average=not train, momentum=0.9,
                     epsilon=1e-5)(x)
    x = nn.avg_pool(x, window_shape=(8,8), padding="SAME")
    x = x.reshape(x.shape[:-3] + (-1,))  # flatten
    x = nn.Dense(self.num_classes)(x)
    return x


# For completeness, we also allow Anderson acceleration for solving
# the implicit differentiation linear system occurring in the backward pass.
def solve_linear_system_fixed_point(matvec, v):
  """Solve linear system matvec(u) = v.

  The solution u* of the system is the fixed point of:
    T(u) = matvec(u) + u - v
  """
  def fixed_point_fun(u):
    d_1_T_transpose_u = tree_add(matvec(u), u)
    return tree_sub(d_1_T_transpose_u, v)

  aa = AndersonAcceleration(fixed_point_fun,
                            history_size=FLAGS.anderson_history_size, tol=1e-2,
                            ridge=FLAGS.anderson_ridge, maxiter=20)
  return aa.run(v)[0]


def main(argv):
  del argv

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  # Solver used for implicit differentiation (backward pass).
  if FLAGS.backward_solver == "normal_cg":
    implicit_solver = partial(solve_normal_cg, tol=1e-2, maxiter=20)
  elif FLAGS.backward_solver == "gmres":
    implicit_solver = partial(solve_gmres, tol=1e-2, maxiter=20)
  elif FLAGS.backward_solver == "anderson":
    implicit_solver = solve_linear_system_fixed_point

  # Solver used for fixed point resolution (forward pass).
  if FLAGS.forward_solver == "anderson":
    fixed_point_solver = partial(AndersonAcceleration,
                                 history_size=FLAGS.anderson_history_size,
                                 ridge=FLAGS.anderson_ridge,
                                 maxiter=FLAGS.forward_maxiter,
                                 tol=FLAGS.forward_tol, implicit_diff=True,
                                 implicit_diff_solve=implicit_solver)
  else:
    fixed_point_solver = partial(FixedPointIteration,
                                 maxiter=FLAGS.forward_maxiter,
                                 tol=FLAGS.forward_tol, implicit_diff=True,
                                 implicit_diff_solve=implicit_solver)

  train_ds, ds_info = load_dataset("train", is_training=True,
                                    batch_size=FLAGS.train_batch_size)
  test_ds, _ = load_dataset("test", is_training=False,
                            batch_size=FLAGS.test_batch_size)
  input_shape = (1,) + ds_info.features["image"].shape
  num_classes = ds_info.features["label"].num_classes

  net = FullDEQ(num_classes, 3 * 8 * FLAGS.net_width, 4 * 8 * FLAGS.net_width,
                fixed_point_solver)

  def predict(all_params, images, train=False):
    """Forward pass in the network on the images."""
    x = images.astype(jnp.float32) / 255.
    mutable = ["batch_stats"] if train else False
    return net.apply(all_params, x, train=train, mutable=mutable)

  logistic_loss = jax.vmap(loss.multiclass_logistic_loss)

  def loss_from_logits(params, l2reg, logits, labels):
    sqnorm = tree_l2_norm(params, squared=True)
    mean_loss = jnp.mean(logistic_loss(labels, logits))
    return mean_loss + 0.5 * l2reg * sqnorm

  @jax.jit
  def accuracy_and_loss(params, l2reg, data, aux):
    all_vars = {"params": params, "batch_stats": aux}
    images, labels = data
    logits = predict(all_vars, images)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    loss = loss_from_logits(params, l2reg, logits, labels)
    return accuracy, loss

  @jax.jit
  def loss_fun(params, l2reg, data, aux):
    all_vars = {"params": params, "batch_stats": aux}
    images, labels = data
    logits, net_state = predict(all_vars, images, train=True)
    loss = loss_from_logits(params, l2reg, logits, labels)
    return loss, net_state["batch_stats"]

  def print_evaluation(params, state, l2reg, data, aux):
    # We don't need `data` because we evaluate on the test set.
    del data

    if state.iter_num % FLAGS.evaluation_frequency == 0:
      # Periodically evaluate classification accuracy on test set.
      accuracy, loss = accuracy_and_loss(params, l2reg, data=next(test_ds),
                                         aux=aux)

      print(f"[Step {state.iter_num}] "
            f"Test accuracy: {accuracy:.3f} "
            f"Test loss: {loss:.3f}.")

    return params, state

  # Initialize solver and parameters.
  solver = OptaxSolver(opt=optax.adam(FLAGS.learning_rate),
                       fun=loss_fun,
                       maxiter=FLAGS.maxiter,
                       pre_update=print_evaluation,
                       has_aux=True)

  rng = jax.random.PRNGKey(0)
  init_vars = net.init(rng, jnp.ones(input_shape), train=True)
  params = init_vars["params"]
  batch_stats = init_vars["batch_stats"]
  state = solver.init_state(params)

  for iternum in range(solver.maxiter):
    params, state = solver.update(params=params, state=state,
                                  l2reg=FLAGS.l2reg, data=next(train_ds),
                                  aux=batch_stats)
    batch_stats = state.aux


if __name__ == "__main__":
  app.run(main)
