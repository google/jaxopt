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
Implementation of sparse coding using jaxopt.
=============================================
"""

import functools
from typing import Optional
from typing import Type
from typing import Mapping
from typing import Any
from typing import Callable
from typing import Tuple

from flax import optim
import jax
import jax.numpy as jnp
from jaxopt import projection
from jaxopt import prox
from jaxopt import proximal_gradient


def dictionary_loss(
    codes: jnp.ndarray,
    params: Tuple[jnp.ndarray, jnp.ndarray],
    reconstruction_loss_fun: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = None
    ):
  """Computes reconstruction loss between data and dict/codes using loss fun.

  Args:
    codes: a samples x components jnp.ndarray of codes.
    params: Tuple containing dictionary and data matrix.
    reconstruction_loss_fun: a callable loss(x, y) -> a real number, where
      x and y are either entries, slices or the matrices themselves.
      Set to 1/2 squared L2 norm of difference by default.

  Returns:
    a float, the reconstruction loss.
  """
  if reconstruction_loss_fun is None:
    reconstruction_loss_fun = lambda x, y: 0.5 * jnp.sum((x - y)**2)

  dic, X = params
  X_pred = codes @ dic
  return reconstruction_loss_fun(X, X_pred)


def make_task_driven_dictionary_learner(
    task_loss_fun: Optional[Callable[[Any, Any, Any, Any], float]] = None,
    reconstruction_loss_fun: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                               jnp.ndarray]] = None,
    optimizer_cls: Optional[Type[optim.Optimizer]] = None,
    optimizer_kw: Mapping[str, Any] = None,
    sparse_coding_kw: Mapping[str, Any] = None):
  """Makes a task driven sparse dictionary learning solver.

  Args:
    task_loss_fun: loss as specified on (codes, dict, task_vars, params) that
      supplements the usual reconstruction loss formulation. If None, only
      dictionary learning is carried out, i.e. that term is assumed to be 0.
    reconstruction_loss_fun: entry (or slice-) wise loss function, set to be
      the Frobenius norm, || . - . ||^2 by default.
    optimizer_cls: Optimizer to solve for dictionary and task_vars (if auxiliary
      task is given). Either None, in which case Jaxopt proximal gradient
      (with sphere projection on dictionary) is used, or a flax
      optimizer class specifying projection on the sphere explicitly for dic.
    optimizer_kw: Arguments to be passed to the optimizer class above, or to
      jaxopt proximal gradient descent.
    sparse_coding_kw: Jaxopt arguments to be passed to the proximal descent
      algorithm computing codes, sparse_coding.

  Returns:
    Function to learn dictionary from data, number of components and
      elastic net regularization, using initialization for dictionary,
      parameters for task and task variables initialization.
  """
  def learner(X: jnp.ndarray,
              n_components: int,
              regularization: float,
              elastic_penalty: float,
              dict_init: Optional[jnp.ndarray] = None,
              task_params: jnp.ndarray = None,
              task_vars_init: jnp.ndarray = None):

    return _task_sparse_dictionary_learning(X, n_components, regularization,
                                            elastic_penalty, dict_init,
                                            task_params, task_vars_init,
                                            reconstruction_loss_fun,
                                            task_loss_fun,
                                            optimizer_cls, optimizer_kw,
                                            sparse_coding_kw)

  return learner


def _task_sparse_dictionary_learning(
    X: jnp.ndarray,
    n_components: int,
    regularization: float,
    elastic_penalty: float,
    dict_init: Optional[jnp.ndarray] = None,
    task_params: jnp.ndarray = None,
    task_vars_init: jnp.ndarray = None,
    reconstruction_loss_fun: Callable[[jnp.ndarray, jnp.ndarray],
                                      jnp.ndarray] = None,
    task_loss_fun: Callable[[Any, Any, Any, Any], float] = None,
    optimizer_cls: Optional[Type[optim.Optimizer]] = None,
    optimizer_kw: Mapping[str, Any] = None,
    sparse_coding_kw: Mapping[str, Any] = None):
  """Computes task driven dictionary, w. implicitly defined sparse codes.

  Given a N x d data matrix X, solves a bilevel optimization problem by seeking
  a dictionary dic of size n_components x d such that, defining implicitly
  codes = sparse_coding(dic, (X, regularization, elastic_penalty))
  one has that dic minimizes
  task_loss(codes, dic, task_var, task_params)
  if such as task_loss was passed on. If None, then task_loss is replaced by
  dictionary_loss(codes, (dic, X)).

  Args:
    X: N x d jnp.ndarray, data matrix with N samples of d features.
    n_components: int, number of atoms in dictionary.
    regularization: regularization strength of elastic penalty.
    elastic_penalty: strength of L2 penalty relative to L1.
    task_params: auxiliary parameters to define task loss, typically data.
    dict_init: initialization for dictionary; that returned by SVD by default.
    reconstruction_loss_fun: loss to be applied to compute reconstruction error.
    task_loss_fun: task driven loss for codes and dictionary using task_vars and
      task_params.
    optimizer_cls: flax optimizer class. If None, falls back on jaxopt projected
      gradient (with sphere normalization constraints). If not None, instantiate
      that optimizer.
    optimizer_kw: parameters passed on to optimizer
    sparse_coding_kw: parameters passed on to jaxopt prox gradient solver.

  Returns:
    the n_components x d dictionary solution found by the algorithm, as well as
    codes.
  """

  if dict_init is None:
    _, _, dict_init = jax.scipy.linalg.svd(X, False)
    dict_init = dict_init[:n_components, :]

  has_task = task_loss_fun is not None

  # Loss function, dictionary learning in addition to task driven loss
  def loss_fun(variables, params):
    dic, task_vars = variables
    coding_params, task_params = params
    codes = sparse_coding(
        dic,
        coding_params,
        reconstruction_loss_fun=reconstruction_loss_fun,
        sparse_coding_kw=sparse_coding_kw)

    if has_task:  # if there is a task, drop loss, replace it with proper value
      loss = task_loss_fun(codes, dic, task_vars, task_params)
    else:
      loss = dictionary_loss(codes, (dic, X), reconstruction_loss_fun)
    return loss, codes

  init = (dict_init, task_vars_init)

  optimizer_kw = {} if optimizer_kw is None else optimizer_kw

  proj_sphere = lambda x: jax.vmap(projection.projection_l2_sphere)(x)
  if optimizer_cls is None:
    # If no optimizer, use jaxopt projected gradient descent.

    # Define projection-prox, here normalize each dict atom by its norm.

    prox_vars = lambda dic_vars, par, s : (
        proj_sphere(dic_vars[0]), dic_vars[1])

    solver = proximal_gradient.make_solver_fun(
        fun=loss_fun, prox=prox_vars, has_aux=True,
        init=init, **optimizer_kw)
    dic, task_vars = solver(((X, regularization, elastic_penalty), task_params))

  else:
    maxiter = optimizer_kw.pop('maxiter', 500)  # Pop'd to set loop size.
    optimizer = optimizer_cls(**optimizer_kw)
    optimizer = optimizer.create(init)

    # Use implicit jaxopt gradients to inform optimizer's steps.
    loss_normalized = lambda dic_vars, params: loss_fun(
        (proj_sphere(dic_vars[0]), dic_vars[1]), params)
    grad_fn = jax.value_and_grad(loss_normalized, has_aux=True)

    def train_step(optimizer, params):
      (loss, codes), grad = grad_fn(optimizer.target, params)
      new_optimizer = optimizer.apply_gradient(grad)
      return new_optimizer, loss

    # Training body fun.
    def body_fun(iteration, in_vars):
      del iteration
      optimizer, pars = in_vars
      optimizer, _ = train_step(optimizer, pars)
      return (optimizer, pars)

    init_val = (optimizer, ((X, regularization, elastic_penalty), task_params))

    # Run fori_loop, this will be converted to a scan.
    optimizer, _ = jax.lax.fori_loop(0, maxiter, body_fun, init_val)

    dic, task_vars = optimizer.target
    # Normalize dictionary before returning it.
    dic = proj_sphere(dic)

  if has_task:
    return dic, task_vars
  return dic


def sparse_coding(dic, params, reconstruction_loss_fun=None,
                  sparse_coding_kw=None, codes_init=None):
  """Computes optimal codes for data X given a dictionary dic."""
  sparse_coding_kw = {} if sparse_coding_kw is None else sparse_coding_kw
  loss_fun = functools.partial(dictionary_loss,
                               reconstruction_loss_fun=reconstruction_loss_fun)
  X, regularization, elastic_penalty = params
  n_components, _ = dic.shape
  N, _ = X.shape

  if codes_init is None:
    codes_init = jnp.zeros((N, n_components))

  solver = proximal_gradient.make_solver_fun(
      fun=loss_fun,
      prox=prox.prox_elastic_net,
      init=codes_init,
      **sparse_coding_kw)

  codes = solver(params_fun=(dic, X),
                 params_prox=[regularization, elastic_penalty])
  return codes
