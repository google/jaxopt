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
Multiclass linear SVM (without intercept).
==========================================

This quadratic program can be solved either with OSQP or with block coordinate descent.
  
Reference:  
  
  Crammer, K. and Singer, Y., 2001. On the algorithmic implementation of multiclass kernel-based vector machines.
  Journal of machine learning research, 2(Dec), pp.265-292.
"""

from absl import app
from absl import flags

import jax
import jax.numpy as jnp

from jaxopt import BlockCoordinateDescent
from jaxopt import OSQP
from jaxopt import objective
from jaxopt import projection
from jaxopt import prox

from sklearn import datasets
from sklearn import preprocessing
from sklearn import svm


flags.DEFINE_float("tol", 1e-5, "Tolerance of solvers.")
flags.DEFINE_float("l2reg", 1000., "Regularization parameter. Must be positive.")
flags.DEFINE_integer("num_samples", 20, "Size of train set.")
flags.DEFINE_integer("num_features", 5, "Features dimension.")
flags.DEFINE_integer("num_classes", 3, "Number of classes.")
flags.DEFINE_bool("verbose", False, "Verbosity.")
FLAGS = flags.FLAGS


def multiclass_linear_svm_skl(X, y, l2reg):
  print("Solve multiclass SVM with sklearn.svm.LinearSVC:")
  svc = svm.LinearSVC(loss="hinge", dual=True, multi_class="crammer_singer",
                      C=1.0 / l2reg, fit_intercept=False,
                      tol=FLAGS.tol, max_iter=100*1000).fit(X, y)
  return svc.coef_.T


def multiclass_linear_svm_bcd(X, Y, l2reg):
  print("Block coordinate descent solution:")

  # Set up parameters.
  block_prox = prox.make_prox_from_projection(projection.projection_simplex)
  fun = objective.multiclass_linear_svm_dual
  data = (X, Y)
  beta_init = jnp.ones((X.shape[0], Y.shape[-1])) / Y.shape[-1]

  # Run solver.
  bcd = BlockCoordinateDescent(fun=fun, block_prox=block_prox,
                               maxiter=10*1000, tol=FLAGS.tol)
  sol = bcd.run(beta_init, hyperparams_prox=None, l2reg=FLAGS.l2reg, data=data)
  return sol.params


def multiclass_linear_svm_osqp(X, Y, l2reg):
  # We solve the problem
  #
  #   minimize 0.5/l2reg beta X X.T beta - (1. - Y)^T beta - 1./l2reg (Y^T X) X^T beta
  #   under        beta >= 0
  #         sum_i beta_i = 1
  #
  print("OSQP solution solution:")

  def matvec_Q(X, beta):
    return 1./l2reg * jnp.dot(X, jnp.dot(X.T, beta))

  linear_part = - (1. - Y) - 1./l2reg * jnp.dot(X, jnp.dot(X.T, Y))

  def matvec_A(_, beta):
    return jnp.sum(beta, axis=-1)

  def matvec_G(_, beta):
    return -beta

  b = jnp.ones(X.shape[0])
  h = jnp.zeros_like(Y)

  osqp = OSQP(matvec_Q=matvec_Q, matvec_A=matvec_A, matvec_G=matvec_G, tol=FLAGS.tol, maxiter=10*1000)
  hyper_params = dict(params_obj=(X, linear_part),
                      params_eq=(None, b),
                      params_ineq=(None, h))
  
  init_params = osqp._box_osqp.init_params(init_x=None, **hyper_params)
  sol, _ = osqp.run(init_params=None, **hyper_params)
  return sol.primal


def main(argv):
  del argv

  # Generate data.
  num_samples = FLAGS.num_samples
  num_features = FLAGS.num_features
  num_classes = FLAGS.num_classes

  X, y = datasets.make_classification(n_samples=num_samples, n_features=num_features,
                                      n_informative=3, n_classes=num_classes, random_state=0)
  X = preprocessing.Normalizer().fit_transform(X)
  Y = preprocessing.LabelBinarizer().fit_transform(y)
  Y = jnp.array(Y)

  l2reg = FLAGS.l2reg

  # Compare against sklearn.
  W_osqp = multiclass_linear_svm_osqp(X, Y, l2reg)
  W_fit_osqp = jnp.dot(X.T, (Y - W_osqp)) / l2reg
  print(W_fit_osqp)
  print()

  W_bcd = multiclass_linear_svm_bcd(X, Y, l2reg)
  W_fit_bcd  = jnp.dot(X.T, (Y - W_bcd)) / l2reg
  print(W_fit_bcd)
  print()

  W_skl = multiclass_linear_svm_skl(X, y, l2reg)
  print(W_skl)
  print()


if __name__ == "__main__":
  jax.config.update("jax_platform_name", "cpu")
  app.run(main)
