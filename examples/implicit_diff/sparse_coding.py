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
Sparse coding.
==============
"""

from absl import app
from absl import flags

import functools
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional

import unittest

import jax
import jax.numpy as jnp
from jax.nn import softplus

from jaxopt import loss
from jaxopt import OptaxSolver
from jaxopt import projection
from jaxopt import prox
from jaxopt import ProximalGradient

import optax
from sklearn import datasets


flags.DEFINE_integer("num_examples", 74, "NUmber of examples.")
flags.DEFINE_integer("num_components", 7, "Number of atoms in dictionnary.")
flags.DEFINE_integer("num_features", 13, "Number of features.")
flags.DEFINE_integer("sparse_coding_maxiter", 100, "Number of iterations for sparse coding.")
flags.DEFINE_integer("maxiter", 10, "Number of iterations of the outer loop.")
flags.DEFINE_float("elastic_penalty", 0.01, "Strength of L2 penalty relative to L1.")
flags.DEFINE_float("regularization", 0.01, "Regularization strength of elastic penalty.")
flags.DEFINE_enum("reconstruction_loss", "squared", ["squared", "abs", "hubert"], "Loss used to build dictionnary.")
FLAGS = flags.FLAGS


def dictionary_loss(
    codes: jnp.ndarray,
    dictionary: jnp.ndarray,
    data: jnp.ndarray,
    reconstruction_loss_fun: Callable[[jnp.ndarray, jnp.ndarray],
                                      jnp.ndarray] = None):
  """Computes reconstruction loss between data and dict/codes using loss fun.

  Args:
    codes: a n_samples x components array of codes.
    dictionary: a components x dimension array
    data: a n_samples x dimension array
    reconstruction_loss_fun: a callable loss(x, y) -> a real number, where
      x and y are either entries, slices or the matrices themselves.
      Set to 1/2 squared L2 norm of difference by default.

  Returns:
    a float, the reconstruction loss.
  """
  if reconstruction_loss_fun is None:
    reconstruction_loss_fun = lambda x, y: 0.5 * jnp.sum((x - y)**2)
  pred = codes @ dictionary
  return reconstruction_loss_fun(data, pred)


def make_task_driven_dictionary_learner(
    task_loss_fun: Optional[Callable[[Any, Any, Any, Any], float]] = None,
    reconstruction_loss_fun: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                               jnp.ndarray]] = None,
    optimizer = None,
    sparse_coding_kw: Mapping[str, Any] = None,
    **kwargs):
  """Makes a task-driven sparse dictionary learning solver.

  Returns a jaxopt solver, using either an optax optimizer or jaxopt prox
  gradient optimizer, to compute, given data, a dictionary whose corresponding
  codes minimizes a given task loss. The solver is defined through the task loss
  function, a reconstruction loss function, and an optimizer. Additional
  parameters can be passed on to lower level functions, notably the computation
  of sparse codes and optimizer parameters.

  Args:
    task_loss_fun: loss as specified on (codes, dict, task_vars, params) that
      supplements the usual reconstruction loss formulation. If None, only
      dictionary learning is carried out, i.e. that term is assumed to be 0.
    reconstruction_loss_fun: entry (or slice-) wise loss function, set to be
      the Frobenius norm between matrices, || . - . ||^2 by default.
    optimizer: optax optimizer. fall back on jaxopt proxgrad if None.
    sparse_coding_kw: Jaxopt arguments to be passed to the proximal descent
      algorithm computing codes, sparse_coding.
    **kwargs: passed onto _task_sparse_dictionary_learning function.

  Returns:
    Function to learn dictionary from data, number of components and
      elastic net regularization, using initialization for dictionary,
      parameters for task and task variables initialization.
  """
  def learner(data: jnp.ndarray,
              n_components: int,
              regularization: float,
              elastic_penalty: float,
              task_vars_init: jnp.ndarray = None,
              task_params: jnp.ndarray = None,
              dic_init: Optional[jnp.ndarray] = None):

    return _task_sparse_dictionary_learning(data, n_components, regularization,
                                            elastic_penalty, task_vars_init,
                                            optimizer,
                                            dic_init, task_params,
                                            reconstruction_loss_fun,
                                            task_loss_fun,
                                            sparse_coding_kw, **kwargs)

  return learner


def _task_sparse_dictionary_learning(
    data: jnp.ndarray,
    n_components: int,
    regularization: float,
    elastic_penalty: float,
    task_vars_init: jnp.ndarray,
    optimizer=None,
    dic_init: Optional[jnp.ndarray] = None,
    task_params: jnp.ndarray = None,
    reconstruction_loss_fun: Callable[[jnp.ndarray, jnp.ndarray],
                                      jnp.ndarray] = None,
    task_loss_fun: Callable[[Any, Any, Any, Any], float] = None,
    sparse_coding_kw: Mapping[str, Any] = None,
    maxiter: int = 100):
  r"""Computes task driven dictionary, w. implicitly defined sparse codes.

  Given a N x d ``data`` matrix, solves a bilevel optimization problem by
  seeking a dictionary ``dic`` of size ``n_components`` x ``d`` such that,
  defining implicitly
  ``codes = sparse_coding(dic, (data, regularization, elastic_penalty))``
  one has that ``dic`` minimizes
  ``task_loss(codes, dic, task_var, task_params)``,
  if such as ``task_loss`` was passed on. If ``task_loss`` is ``None``, then
  ``task_loss`` is replaced by default by
  ``dictionary_loss(codes, (dic, data))``.

  Args:
    data: N x d jnp.ndarray, data matrix with N samples of d features.
    n_components: int, number of atoms in dictionary.
    regularization: regularization strength of elastic penalty.
    elastic_penalty: strength of L2 penalty relative to L1.
    task_vars_init: initializer for task related optimization variables.
    optimizer: If None, falls back on jaxopt proximal gradient (with sphere
      projection for ``dic``). If not ``None``, use that algorithm's method with
      a normalized dictionary.
    dic_init: initialization for dictionary; that returned by SVD by default.
    reconstruction_loss_fun: loss to be applied to compute reconstruction error.
    task_params: auxiliary parameters to define task loss, typically data.
    task_loss_fun: task driven loss for codes and dictionary using task_vars and
      task_params.
    sparse_coding_kw: parameters passed on to jaxopt prox gradient solver to
      compute codes.
    maxiter: maximal number of iterations of the outer loop.

  Returns:
    A``n_components x d`` matrix, the ``dic`` solution found by the algorithm,
    as well as task variables if task was provided.
  """

  if dic_init is None:
    _, _, dic_init = jax.scipy.linalg.svd(data, False)
    dic_init = dic_init[:n_components, :]

  has_task = task_loss_fun is not None

  # Loss function, dictionary learning in addition to task driven loss
  def loss_fun(params, hyper_params):
    dic, task_vars = params
    coding_params, task_params = hyper_params
    codes = sparse_coding(
        dic,
        coding_params,
        reconstruction_loss_fun=reconstruction_loss_fun,
        sparse_coding_kw=sparse_coding_kw)
    if optimizer is not None:
      dic = projection.projection_l2_sphere(dic)

    if has_task:
      loss = task_loss_fun(codes, dic, task_vars, task_params)
    else:
      loss = dictionary_loss(codes, dic, data, reconstruction_loss_fun)
    return loss, codes

  def prox_dic(params, hyper, step):
    # Here projection/prox is only applied on the dictionary.
    del hyper, step
    dic, task_vars = params
    return projection.projection_l2_sphere(dic), task_vars

  if optimizer is None:
    solver = ProximalGradient(fun=loss_fun, prox=prox_dic, has_aux=True)
    params = (dic_init, task_vars_init)
    state = solver.init_state(params, None)

    for _ in range(maxiter):
      params, state = solver.update(
          params, state, None,
          ((data, regularization, elastic_penalty), task_params))

      # Normalize dictionary before returning it.
      dic, task_vars = prox_dic(params, None, None)

  else:
    solver = OptaxSolver(opt=optimizer, fun=loss_fun, has_aux=True)
    params = (dic_init, task_vars_init)
    state = solver.init_state(params)

    for _ in range(maxiter):
      params, state = solver.update(
          params, state,
          ((data, regularization, elastic_penalty), task_params))

      # Normalize dictionary before returning it.
      dic, task_vars = prox_dic(params, None, None)

  if has_task:
    return dic, task_vars
  return dic


def sparse_coding(dic, params, reconstruction_loss_fun=None,
                  sparse_coding_kw=None, codes_init=None):
  """Computes optimal codes for data given a dictionary dic using params."""
  sparse_coding_kw = {} if sparse_coding_kw is None else sparse_coding_kw
  loss_fun = functools.partial(dictionary_loss,
                               reconstruction_loss_fun=reconstruction_loss_fun)
  data, regularization, elastic_penalty = params
  n_components, _ = dic.shape
  n_points, _ = data.shape

  if codes_init is None:
    codes_init = jnp.zeros((n_points, n_components))

  solver = ProximalGradient(
      fun=loss_fun,
      prox=prox.prox_elastic_net,
      **sparse_coding_kw)

  codes = solver.run(codes_init, [regularization, elastic_penalty],
                     dic, data).params
  return codes


def main(argv):
  del argv

  # needed for asserts
  tc = unittest.TestCase()

  N = FLAGS.num_examples
  k = FLAGS.num_components
  d = FLAGS.num_features
  # X is N x d
  # dic is k x d
  X, dictionary_0, codes_0 = datasets.make_sparse_coded_signal(
      n_samples=N,
      n_components=k,
      n_features=d,
      n_nonzero_coefs=k//2,
      random_state=0,
  )
  X = X.T # bug in https://github.com/scikit-learn/scikit-learn/issues/19894
  X = .1 * X + .0001 * jax.random.normal(jax.random.PRNGKey(0), (N, d))

  if FLAGS.reconstruction_loss == "squared":
    reconstruction_loss_fun = None
  elif FLAGS.reconstruction_loss == "abs":
    reconstruction_loss_fun = lambda x, y: jnp.sum(jnp.abs(x - y)**2.1)
  elif FLAGS.reconstruction_loss == "hubert":
    reconstruction_loss_fun = lambda x, y: jnp.sum(loss.huber_loss(x, y, .01))
  else:
    raise ValueError(f"Unkwown reconstruction_loss {FLAGS.reconstruction_loss}")

  elastic_penalty = FLAGS.elastic_penalty
  regularization = FLAGS.regularization

  # slightly complicated Vanilla dictionary learning when no task.
  # complicated in the sense that Danskin is not used. Here using prox from
  # jaxopt.
  solver = jax.jit(
      make_task_driven_dictionary_learner(
          reconstruction_loss_fun=reconstruction_loss_fun, maxiter=FLAGS.maxiter,
          sparse_coding_kw={'maxiter': FLAGS.sparse_coding_maxiter}),
      static_argnums=(1, 8))  # n_components & reconstruction_loss_fun

  print("Create dictionnary with no task:", flush=True)
  # Compute dictionary
  dic_jop_0 = solver(
      X,
      n_components=k,
      regularization=regularization,
      elastic_penalty=elastic_penalty)
  tc.assertEqual(dic_jop_0.shape, (k, d))
  print(dic_jop_0)

  # Test now task driven dictionary learning using *arbitrary* labels computed
  # from initial codes. This is a binary logistic regression problem.
  label = jnp.sum(codes_0[0:3, :], axis=0) > 0
  def task_loss_fun(codes, dic, task_vars, task_params):
    del dic
    w, b = task_vars
    logit = jnp.dot(codes, w) + b
    return jnp.sum(
        jnp.sum(softplus(logit) - label * logit) + 0.5 * task_params *
        (jnp.dot(w, w) + b * b))

  # Create a solver that will now use optax's Adam to learn both dic and
  # logistic regression parameters.
  solver = jax.jit(
      make_task_driven_dictionary_learner(
          task_loss_fun=task_loss_fun,
          reconstruction_loss_fun=reconstruction_loss_fun, maxiter=FLAGS.maxiter,
          sparse_coding_kw={'maxiter': FLAGS.sparse_coding_maxiter},
          optimizer=optax.adam(1e-3)),
      static_argnums=(1, 8))  # n_components & reconstruction_loss_fun

  print("Compute task driven dictionnary:", flush=True)
  dic_jop_task, w_and_b = solver(
      X,
      n_components=k,
      regularization=regularization,
      elastic_penalty=elastic_penalty,
      task_vars_init=(jnp.zeros(k), jnp.zeros(1)),
      task_params=0.001)
  print(dic_jop_task)

  # Check we have at least improved results using the very same w_and_b
  losses = []
  for dic in [dic_jop_0, dic_jop_task]:
    losses.append(
        task_loss_fun(
            sparse_coding(
                dic, (X, regularization, elastic_penalty)), dic, w_and_b,
            0.0))
  tc.assertGreater(losses[0], losses[1])
  print(f"With task the loss ({losses[1]}) is smaller than without task ({losses[0]})")


if __name__ == "__main__":
  app.run(main)
