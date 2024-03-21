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


FLAGS = flags.FLAGS

flags.DEFINE_integer("maxiter", default=30, help="Max # of iterations.")
flags.DEFINE_integer("n_samples", default=10000, help="Number of samples.")
flags.DEFINE_integer("n_features", default=200, help="Number of features.")
flags.DEFINE_string("task", "binary_logreg", "Task to benchmark.")


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
  else:
    raise ValueError("Invalid task name.")


if __name__ == '__main__':
  app.run(main)
