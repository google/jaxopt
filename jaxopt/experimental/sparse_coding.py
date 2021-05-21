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

"""Implementation of sparse coding using jaxopt."""

import functools
from typing import Optional
from typing import Type

from flax import optim
import jax
import jax.numpy as jnp
from jaxopt import projection
from jaxopt import prox
from jaxopt import proximal_gradient


def dictionary_loss(codes, params):
  """Computes quadratic reconstruction loss between data and dict/codes."""
  dic, X = params
  X_pred = codes @ dic
  return 0.5 * jnp.sum((X_pred - X) ** 2)


@functools.partial(jax.jit, static_argnums=(1, 5, 6, 7, 8))
def sparse_dictionary_learning(X: jnp.ndarray,
                  n_components: int,
                  regularization: float,
                  elastic_penalty: float = 0.0,
                  dict_init: Optional[jnp.ndarray] = None,
                  optimizer_cls: Optional[Type[optim.Optimizer]] = None,
                  maxiter: int = 100,
                  verbose: int = 0,
                  learning_rate: float = 0.0):
  """Computes optimal dictionary, using implicitly defined sparse codes.

  Given a N x d data matrix X, solves a bilevel optimization problem by seeking
  a dictionary dic of size n_components x d such that
  dictionary_loss(sparse_coding(dic, (X, params), (dic, X)) is small,
  where sparse_coding returns optimal codes fitted to X using dic and
  regularization parameters.

  Args:
    X: N x d jnp.ndarray, data matrix with N samples of d features.
    n_components: int, number of atoms in dictionary.
    regularization: regularization strength of elastic penalty.
    elastic_penalty: strength of L2 penalty relative to L1.
    dict_init: initialization for dictionary; that returned by SVD by default.
    optimizer_cls: flax optimizer class. If None, falls back on jaxopt projected
      gradient (with sphere normalization constraints). If not None, instantiate
      that optimizer.
    maxiter: max number of iterations when learning the dictionary.
    learning_rate: use to initialize optimizer_cls or stepsize in prox gradient.

  Returns:
    the n_components x d dictionary solution found by the algorithm.
  """

  if dict_init is None:
    _, _, dict_init = jax.scipy.linalg.svd(X, False)
    dict_init = dict_init[:n_components, :]

  if optimizer_cls is None:
    # If no optimizer, fall back on jaxopt projected gradient descent.

    # Define projection-prox, here normalize each dict atom by its norm.
    prox_sphere = lambda x, par, s : jax.vmap(
        projection.projection_l2_sphere)(x)

    solver = proximal_gradient.make_solver_fun(
        fun=codes_from_dictionary, prox=prox_sphere,
        init=dict_init, verbose=verbose, has_aux=True,
        maxiter=maxiter, stepsize=learning_rate)
    return solver((X, regularization, elastic_penalty))

  else:
    optimizer = optimizer_cls(
        learning_rate=jnp.where(learning_rate > 0.0, learning_rate, 1e-3))
    optimizer = optimizer.create(dict_init)

    # Use implicit jaxopt gradients to inform optimizer's steps.
    def train_step(optimizer, params):
      codes_from_dic_normalized = lambda dic, params: codes_from_dictionary(
          jax.vmap(projection.projection_l2_sphere)(dic), params)
      grad_fn = jax.value_and_grad(codes_from_dic_normalized,
                                   has_aux=True)
      (loss, aux), grad = grad_fn(optimizer.target,
                                  params)
      new_optimizer = optimizer.apply_gradient(grad)
      return new_optimizer, loss, aux, jnp.mean(grad ** 2)

    # Training body fun.
    def body_fun(iteration, in_vars):
      optimizer, pars = in_vars
      optimizer, loss, _, grad_norm = train_step(optimizer, pars)
      if verbose:
        print(iteration, loss, grad_norm)
      return (optimizer, pars)

    init_val = (optimizer, (X, regularization, elastic_penalty))

    # Run fori_loop, this will be converted to a scan.
    optimizer, _ = jax.lax.fori_loop(0, maxiter, body_fun, init_val)

    return jax.vmap(projection.projection_l2_sphere)(optimizer.target)


def sparse_coding(dic, params):
  """Computes optimal codes for data X given a dictionary dic."""
  X, regularization, elastic_penalty = params
  n_components, _ = dic.shape
  N, _ = X.shape
  solver = proximal_gradient.make_solver_fun(
      fun=dictionary_loss,
      prox=prox.prox_elastic_net,
      init=jnp.zeros((N, n_components)))
  codes = solver(params_fun=(dic, X),
                 params_prox=[regularization, elastic_penalty])
  return dictionary_loss(codes, (dic, X)) , codes
