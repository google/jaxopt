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

"""Benchmark JAX implementation vs. NumPy implementation of proximal gradient."""

import time

from typing import NamedTuple
from typing import Sequence

from absl import app
from absl import flags

from sklearn import datasets
from sklearn import preprocessing

import numpy as onp

import jax
import jax.numpy as jnp

from jaxopt.proximal_gradient import proximal_gradient as proximal_gradient_jaxopt
from jaxopt.prox import prox_lasso


FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", default="boston", help=("Dataset to use."))
flags.DEFINE_bool("float64", default=False, help=("Enable double precision."))
flags.DEFINE_float("lam", default=1.0, help=("Regularization value."))
flags.DEFINE_integer("maxiter", default=200, help=("Max # of iterations."))
flags.DEFINE_integer("n_samples", default=100000, help=("Number of samples."))
flags.DEFINE_integer("n_features", default=200, help=("Number of features."))
flags.DEFINE_bool("verbose", default=False, help=("Enable verbose output."))


class OptimizeResults(NamedTuple):
  error: float
  nit: int
  x: onp.ndarray


def _make_linesearch(fun, prox, maxls):
  def linesearch(curr_x, curr_x_fun_val, curr_x_fun_grad, curr_stepsize):
    """A pure NumPy re-implementation of linesearch for benchmarking reasons."""
    for _ in range(maxls):
      next_x = prox(curr_x - curr_stepsize * curr_x_fun_grad, curr_stepsize)
      diff = next_x - curr_x
      sqdist = onp.vdot(diff, diff)
      value_F = fun(next_x)
      value_Q = (curr_x_fun_val + onp.vdot(diff, curr_x_fun_grad) +
                 0.5 * sqdist / curr_stepsize)
      if value_F <= value_Q:
        return next_x, curr_stepsize
      curr_stepsize *= 0.5

    # Undo the last decrase when `maxls` is reached.
    curr_stepsize *= 2

    return next_x, curr_stepsize
  return linesearch


def proximal_gradient_onp(fun, init, prox, stepsize, maxiter=200, maxls=15, tol=1e-3,
             verbose=0):
  """A pure NumPy re-implementation of proximal gradient for benchmarking."""
  curr_x = init
  curr_stepsize = 1.0
  linesearch = _make_linesearch(fun, prox, maxls)

  for iter_num in range(1, maxiter + 1):
    # Convergence monitoring.
    curr_x_fun_val, curr_x_fun_grad = fun(curr_x, grad=True)
    diff_x = curr_x - prox(curr_x - curr_x_fun_grad)
    curr_error = onp.sqrt(onp.sum(diff_x ** 2))
    if verbose: print(iter_num, curr_error)
    if curr_error <= tol: break

    if stepsize <= 0:
      # With line search.
      curr_x, curr_stepsize = linesearch(curr_x, curr_x_fun_val,
                                         curr_x_fun_grad, curr_stepsize)
      if curr_stepsize <= 1e-6:
        # Reset step size.
        curr_stepsize = 1.0
      else:
        curr_stepsize *= 2
    else:
      # Without line search.
      curr_x = prox(curr_x - stepsize * curr_x_fun_grad, stepsize)

  return OptimizeResults(x=curr_x, nit=iter_num, error=curr_error)


def proximal_gradient_accel_onp(fun, init, prox, stepsize, maxiter=200, maxls=15, tol=1e-3,
              verbose=0):
  """A pure NumPy re-implementation of proximal gradient with acceleration."""
  curr_x = init
  curr_y = init
  curr_t = 1.0
  curr_stepsize = 1.0
  linesearch = _make_linesearch(fun, prox, maxls)

  for iter_num in range(1, maxiter + 1):
    # Convergence monitoring
    curr_x_fun_grad = fun(curr_x, grad=True)[1]
    diff_x = curr_x - prox(curr_x - curr_x_fun_grad)
    curr_error = onp.sqrt(onp.sum(diff_x ** 2))
    if verbose: print(iter_num, curr_error)
    if curr_error <= tol: break

    # Iteration.
    curr_y_fun_val, curr_y_fun_grad = fun(curr_y, grad=True)

    if stepsize <= 0:
      # With line search.
      next_x, curr_stepsize = linesearch(curr_y, curr_y_fun_val,
                                         curr_y_fun_grad, curr_stepsize)
      if curr_stepsize <= 1e-6:
        # Reset step size.
        curr_stepsize = 1.0
      else:
        curr_stepsize *= 2
    else:
      # Without line search.
      next_x = prox(curr_y - stepsize * curr_y_fun_grad, stepsize)

    next_t = 0.5 * (1 + onp.sqrt(1 + 4 * curr_t ** 2))
    diff_x = next_x - curr_x
    next_y = next_x + (curr_t - 1) / next_t * diff_x
    curr_x = next_x
    curr_y = next_y
    curr_t = next_t

  return OptimizeResults(x=curr_x, nit=iter_num, error=curr_error)


def lasso_onp(X, y, lam, stepsize, tol, maxiter, acceleration, verbose):
  def fun(w, grad=False):
    y_pred = onp.dot(X, w)
    diff = y_pred - y
    obj = 0.5 * onp.dot(diff, diff)
    if not grad: return obj
    g = onp.dot(X.T, diff)
    return obj, g

  def prox(w, stepsize=1.0):
    return onp.sign(w) * onp.maximum(onp.abs(w) - lam * stepsize, 0)

  init = onp.zeros(X.shape[1], dtype=X.dtype)
  solver_fun = proximal_gradient_accel_onp if acceleration else proximal_gradient_onp
  return solver_fun(fun=fun, init=init, prox=prox, stepsize=stepsize,
                    maxiter=maxiter, tol=tol, verbose=verbose)


def lasso_jnp(X, y, lam, stepsize, tol, maxiter, acceleration, verbose):
  def fun(w, _):
    y_pred = jnp.dot(X, w)
    diff = y_pred - y
    return 0.5 * jnp.dot(diff, diff)

  init = jnp.zeros(X.shape[1], dtype=X.dtype)
  return proximal_gradient_jaxopt(fun=fun, init=init, prox=prox_lasso, params_prox=lam,
                      tol=tol, stepsize=stepsize, maxiter=maxiter,
                      acceleration=acceleration, verbose=verbose, ret_info=True)


def run_proximal_gradient(X, y, lam, stepsize, maxiter, verbose):
  if stepsize <= 0:
    print("proximal gradient (line search)")
  else:
    print("proximal gradient (constant step size)")
  print("-" * 50)
  start = time.time()
  res_onp = lasso_onp(X=X, y=y, lam=lam, stepsize=stepsize, tol=1e-3,
                      maxiter=maxiter, acceleration=False, verbose=verbose)
  print("error onp:", res_onp.error)
  print("iter_num onp:", res_onp.nit)
  print("time onp", time.time() - start)
  print(flush=True)

  start = time.time()
  res_jnp = lasso_jnp(X=X, y=y, lam=lam, stepsize=stepsize, tol=1e-3,
                      maxiter=maxiter, acceleration=False, verbose=verbose)
  print("error jnp:", res_jnp.error)
  print("iter_num jnp:", res_jnp.nit)
  print("time jnp", time.time() - start)
  print(flush=True)


def run_accelerated_proximal_gradient(X, y, lam, stepsize, maxiter, verbose):
  if stepsize <= 0:
    print("accelerated proximal gradient descent (line search)")
  else:
    print("accelerated proximal gradient descent (constant step size)")
  print("-" * 50)
  start = time.time()
  res_onp = lasso_onp(X=X, y=y, lam=lam, stepsize=stepsize, tol=1e-3,
                      maxiter=maxiter, acceleration=True, verbose=verbose)
  print("error onp:", res_onp.error)
  print("iter_num onp:", res_onp.nit)
  print("time onp", time.time() - start)
  print(flush=True)

  start = time.time()
  res_jnp = lasso_jnp(X=X, y=y, lam=lam, stepsize=stepsize, tol=1e-3,
                      maxiter=maxiter, acceleration=True, verbose=verbose)
  print("error jnp:", res_jnp.error)
  print("iter_num jnp:", res_jnp.nit)
  print("time jnp", time.time() - start)
  print(flush=True)


def load_dataset(dataset, float64=False):
  if dataset == "boston":
    X, y = datasets.load_boston(return_X_y=True)
  elif dataset == "synth":
    X, y = datasets.make_classification(n_samples=FLAGS.n_samples,
                                        n_features=FLAGS.n_features,
                                        n_classes=2,
                                        random_state=0)
  else:
    raise ValueError("Invalid dataset.")

  X = preprocessing.Normalizer().fit_transform(X)

  if not float64:
    X = X.astype(onp.float32)
    y = y.astype(onp.float32)

  return X, y


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.float64:
    jax.config.update("jax_enable_x64", True)

  X, y = load_dataset(FLAGS.dataset, FLAGS.float64)

  print("Dataset:", FLAGS.dataset)
  print("n_samples:", X.shape[0])
  print("n_features:", X.shape[1])
  print("lambda:", FLAGS.lam)
  print("maxiter:", FLAGS.maxiter)
  print("float64:", FLAGS.float64)
  print()

  kw = dict(lam=FLAGS.lam, maxiter=FLAGS.maxiter, verbose=FLAGS.verbose)
  run_proximal_gradient(X, y, stepsize=1e-3, **kw)
  run_proximal_gradient(X, y, stepsize=0, **kw)
  run_accelerated_proximal_gradient(X, y, stepsize=1e-3, **kw)
  run_accelerated_proximal_gradient(X, y, stepsize=0, **kw)


if __name__ == '__main__':
  app.run(main)
