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

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from jaxopt import objective
from jaxopt._src import test_util
from jaxopt import BacktrackingLineSearch
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_negative
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_vdot

import numpy as onp

from sklearn import datasets

N_FUN_CALLS = 0
N_GRAD_CALLS = 0
class BacktrackingLinesearchTest(test_util.JaxoptTestCase):

  def _check_conditions_satisfied(
      self,
      condition_name,
      c1,
      c2,
      stepsize,
      initial_value,
      initial_grad,
      final_state):
    self.assertTrue(jnp.all(final_state.done))
    self.assertFalse(jnp.any(final_state.failed))

    descent_direction = tree_scalar_mul(-1, initial_grad)
    # Check sufficient decrease for all line search methods.
    sufficient_decrease = jnp.all(
        final_state.value <= initial_value +
        c1 * stepsize * tree_vdot(final_state.grad, descent_direction))
    self.assertTrue(sufficient_decrease)

    if condition_name == "goldstein":
      goldstein = jnp.all(
          final_state.value >= initial_value +
          (1 - c1) * stepsize * tree_vdot(
              initial_grad, descent_direction))
      self.assertTrue(goldstein)
    elif condition_name == "strong-wolfe":
      new_slope = tree_vdot(final_state.grad, descent_direction)
      slope = tree_vdot(initial_grad, descent_direction)
      curvature = jnp.all(jnp.abs(new_slope) <= c2 * jnp.abs(slope))
      self.assertTrue(curvature)

    elif condition_name == "wolfe":
      new_slope = tree_vdot(final_state.grad, descent_direction)
      slope = tree_vdot(initial_grad, descent_direction)
      curvature = jnp.all(new_slope >= c2 * slope)
      self.assertTrue(curvature)

  def _check_stepsize_match_params(self,
                                   value_and_grad_fun, 
                                   stepsize, 
                                   params, 
                                   descent_direction, 
                                   state, 
                                   **fun_kwargs):
    self.assertTrue(state.failed or state.done)
    step = tree_add_scalar_mul(params, stepsize, descent_direction)
    value, grad = value_and_grad_fun(step, **fun_kwargs)
    self.assertAllClose(step, state.params)
    self.assertAllClose(value, state.value)
    self.assertAllClose(grad, state.grad)


  def test_aux_value(self):
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=2,
                                        n_informative=3, random_state=0)
    data = (X, y)
    true_fun = objective.binary_logreg

    def augmented_fun(w, data):
      value = true_fun(w, data)
      aux = w[0:1]
      return value, aux

    rng = onp.random.RandomState(0)
    w_init = rng.randn(X.shape[1])
    ls = BacktrackingLineSearch(
        fun=augmented_fun, maxiter=20, condition="armijo", has_aux=True
    )
    _, state = ls.run(
        init_stepsize=1.0, params=w_init, fun_kwargs={"data": data}
    )
    _, aux_at_params = augmented_fun(state.params, data)
    self.assertArraysEqual(aux_at_params, state.aux)

  @parameterized.product(cond=[
      "armijo", "goldstein", "strong-wolfe", "wolfe"])
  def test_backtracking_linesearch(self, cond):
    X, y = datasets.make_classification(
        n_samples=10, n_features=5, n_classes=2, n_informative=3, random_state=0
    )
    data = (X, y)
    fun = objective.binary_logreg

    rng = onp.random.RandomState(0)
    w_init = rng.randn(X.shape[1])
    initial_grad = jax.grad(fun)(w_init, data=data)
    initial_value = fun(w_init, data=data)

    # Manual loop.
    ls = BacktrackingLineSearch(fun=fun, condition=cond)
    stepsize = 1.0
    state = ls.init_state(
        init_stepsize=1.0, params=w_init, fun_kwargs={"data": data}
    )
    stepsize, state = ls.update(stepsize=stepsize, state=state, params=w_init,
                                fun_kwargs={"data": data})
    error1 = state.error
    _, state = ls.update(
        stepsize=stepsize, state=state, params=w_init, fun_kwargs={"data": data}
    )
    error2 = state.error
    self.assertLessEqual(error2, error1)

    # Call to run.
    ls = BacktrackingLineSearch(fun=fun, maxiter=20, condition=cond)
    stepsize, state = ls.run(
        init_stepsize=1.0, params=w_init, fun_kwargs={"data": data}
    )
    self.assertLessEqual(state.error, 1e-5)
    self._check_conditions_satisfied(
        cond, ls.c1, ls.c2, stepsize, initial_value, initial_grad, state)
    self._check_stepsize_match_params(jax.value_and_grad(fun),
                                      stepsize,
                                      w_init,
                                      tree_negative(initial_grad),
                                      state, 
                                      data=data)

    # Call to run with value_and_grad=True.
    ls = BacktrackingLineSearch(fun=jax.value_and_grad(fun),
                                condition=cond,
                                maxiter=20,
                                value_and_grad=True)
    stepsize, state = ls.run(
        init_stepsize=1.0, params=w_init, fun_kwargs={"data": data}
    )
    self.assertLessEqual(state.error, 1e-5)
    self._check_conditions_satisfied(
        cond, ls.c1, ls.c2, stepsize, initial_value, initial_grad, state)
    self._check_stepsize_match_params(jax.value_and_grad(fun),
                                      stepsize,
                                      w_init,
                                      tree_negative(initial_grad),
                                      state,
                                      data=data)
    # Failed linesearch (high c1 ensures convergence condition is not met).
    ls = BacktrackingLineSearch(fun=fun, maxiter=20, condition=cond, c1=2.)
    stepsize, state = ls.run(
        init_stepsize=1.0, params=w_init, fun_kwargs={"data": data}
    )
    self.assertTrue(jnp.all(state.failed))
    self.assertFalse(jnp.any(state.done))
    self._check_stepsize_match_params(jax.value_and_grad(fun),
                                      stepsize,
                                      w_init,
                                      tree_negative(initial_grad),
                                      state,
                                      data=data)

  @parameterized.product(
      cond=["armijo", "goldstein", "strong-wolfe", "wolfe"],
      val=[onp.inf, onp.nan])
  def test_backtracking_linesearch_non_finite(self, cond, val):

    def fun(x):
      result = jnp.where(x > 4., val, (x - 2)**2)
      grad = jnp.where(x > 4., onp.nan, 2 * (x - 2.))
      return result, grad

    x_init = -0.001
    _, grad = fun(x_init)
    descent_direction = tree_negative(grad)

    # Make sure initial step triggers a non-finite value.
    ls = BacktrackingLineSearch(
        fun=fun, condition=cond, max_stepsize=2.0, value_and_grad=True)
    stepsize = 1.25

    stepsize, state = ls.run(init_stepsize=stepsize, params=x_init)
    self._check_stepsize_match_params(fun,
                                      stepsize,
                                      x_init,
                                      descent_direction,
                                      state)
    self.assertFalse(jnp.isnan(state.value) or jnp.isinf(state.value))

  @parameterized.product(cond=["armijo", "goldstein"])
  def test_eval_count(self, cond):
    global N_FUN_CALLS
    global N_GRAD_CALLS

    def fun(x):
      global N_FUN_CALLS
      N_FUN_CALLS += 1
      return 32*jnp.sum(x**2)
    
    _value_and_grad_fun = jax.value_and_grad(fun)

    def value_and_grad_fun(x):
      global N_GRAD_CALLS
      N_GRAD_CALLS += 1
      return _value_and_grad_fun(x)
    
    params = jnp.asarray(1.)
    init_stepsize = jnp.asarray(1.)
    # Reset number of calls
    N_FUN_CALLS = 0
    N_GRAD_CALLS = 0

    linesearch = BacktrackingLineSearch(fun,
                                        value_and_grad=value_and_grad_fun,
                                        condition=cond,
                                        jit=False)
    _, ls_state = linesearch.run(init_stepsize, params)
    self.assertEqual(N_FUN_CALLS, ls_state.num_fun_eval)
    self.assertEqual(N_GRAD_CALLS, ls_state.num_grad_eval)
    # One initial gradient call and one final = 2 calls
    self.assertEqual(N_GRAD_CALLS, 2)
    self.assertGreater(N_FUN_CALLS, N_GRAD_CALLS)


if __name__ == '__main__':
  # Uncomment the line below in order to run in float64.
  #jax.config.update("jax_enable_x64", True)
  absltest.main()
