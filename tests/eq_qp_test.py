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

from jaxopt import projection
from jaxopt.base import KKTSolution
from jaxopt import EqualityConstrainedQP
from jaxopt._src import test_util
from jaxopt._src.tree_util import tree_negative

import numpy as onp


class EqualityConstrainedQPTest(test_util.JaxoptTestCase):
  def _check_derivative_Q_c_A_b(self, solver, params, Q, c, A, b):
    def fun(Q, c, A, b):
      Q = 0.5 * (Q + Q.T)

      hyperparams = dict(params_obj=(Q, c), params_eq=(A, b))

      # reduce the primal variables to a scalar value for test purpose.
      return jnp.sum(solver.run(**hyperparams).params[0])

    # Derivative w.r.t. A.
    rng = onp.random.RandomState(0)
    V = rng.rand(*A.shape)
    V /= onp.sqrt(onp.sum(V ** 2))
    eps = 1e-4
    deriv_jax = jnp.vdot(V, jax.grad(fun, argnums=2)(Q, c, A, b))
    deriv_num = (fun(Q, c, A + eps * V, b) - fun(Q, c, A - eps * V, b)) / (2 * eps)
    self.assertAllClose(deriv_jax, deriv_num, atol=1e-3)

    # Derivative w.r.t. b.
    v = rng.rand(*b.shape)
    v /= onp.sqrt(onp.sum(v ** 2))
    eps = 1e-4
    deriv_jax = jnp.vdot(v, jax.grad(fun, argnums=3)(Q, c, A, b))
    deriv_num = (fun(Q, c, A, b + eps * v) - fun(Q, c, A, b - eps * v)) / (2 * eps)
    self.assertAllClose(deriv_jax, deriv_num, atol=1e-3)

    # Derivative w.r.t. Q
    W = rng.rand(*Q.shape)
    W /= onp.sqrt(onp.sum(W ** 2))
    eps = 1e-4
    deriv_jax = jnp.vdot(W, jax.grad(fun, argnums=0)(Q, c, A, b))
    deriv_num = (fun(Q + eps * W, c, A, b) - fun(Q - eps * W, c, A, b)) / (2 * eps)
    self.assertAllClose(deriv_jax, deriv_num, atol=1e-3)

    # Derivative w.r.t. c
    w = rng.rand(*c.shape)
    w /= onp.sqrt(onp.sum(w ** 2))
    eps = 1e-4
    deriv_jax = jnp.vdot(w, jax.grad(fun, argnums=1)(Q, c, A, b))
    deriv_num = (fun(Q, c + eps * w, A, b) - fun(Q, c - eps * w, A, b)) / (2 * eps)
    self.assertAllClose(deriv_jax, deriv_num, atol=1e-3)

  def test_qp_eq_only(self):
    Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
    c = jnp.array([1.0, 1.0])
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([1.0])
    qp = EqualityConstrainedQP(tol=1e-7)
    hyperparams = dict(params_obj=(Q, c), params_eq=(A, b))
    sol = qp.run(**hyperparams).params
    self.assertAllClose(qp.l2_optimality_error(sol, **hyperparams), 0.0)
    self._check_derivative_Q_c_A_b(qp, hyperparams, Q, c, A, b)

  def test_qp_eq_with_init(self):
    Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
    c = jnp.array([1.0, 1.0])
    A = jnp.array([[1.0, 1.0]])
    b = jnp.array([1.0])
    qp = EqualityConstrainedQP(tol=1e-7)
    hyperparams = dict(params_obj=(Q, c), params_eq=(A, b))
    init_params = KKTSolution(jnp.array([1.0, 1.0]), jnp.array([1.0]))
    sol = qp.run(init_params, **hyperparams).params
    self.assertAllClose(qp.l2_optimality_error(sol, **hyperparams), 0.0)
    self._check_derivative_Q_c_A_b(qp, hyperparams, Q, c, A, b)

  def test_projection_hyperplane(self):
    x = jnp.array([1.0, 2.0])
    a = jnp.array([-0.5, 1.5])
    b = 0.3
    # Find ||y-x||^2 such that jnp.dot(y, a) = b.
    expected = projection.projection_hyperplane(x, (a, b))

    matvec_Q = lambda params_Q, u: u
    matvec_A = lambda params_A, u: jnp.dot(a, u).reshape(1)
    qp = EqualityConstrainedQP(matvec_Q=matvec_Q, matvec_A=matvec_A)
    # In this example, params_Q = params_A = None.
    hyperparams = dict(params_obj=(None, -x), params_eq=(None, jnp.array([b])))
    sol = qp.run(**hyperparams).params
    primal_sol = sol[0]
    self.assertArraysAllClose(primal_sol, expected)
    self.assertAllClose(qp.l2_optimality_error(sol, **hyperparams), 0.0)

  def test_pytree_api(self):
    Q1 = jnp.array([[1.0, -0.5],
                  [-0.5, 1.0]])
    Q2 = jnp.array([[2.0]])
    Q = {'problem1': Q1, 'problem2': Q2}

    c1 = jnp.array([-0.4, 0.3])
    c2 = jnp.array([0.1])
    c = {'problem1': c1, 'problem2': c2}

    a1 = jnp.array([[-0.5, 1.5]])
    a2 = jnp.array([[10.0]])
    A = {'problem1': a1, 'problem2': a2}

    b1 = jnp.array([0.3])
    b2 = jnp.array([5.0])
    b = {'problem1': b1, 'problem2': b2}

    qp = EqualityConstrainedQP(tol=1e-3)
    hyperparams = dict(params_obj=(Q, c), params_eq=(A, b))
    # Test that there is no error when running the solver.
    sol = qp.run(**hyperparams).params

  @parameterized.product(refine_regularization=[0., 1e-6])
  def test_matvec_api(self, refine_regularization):
    # refine_regularization != 0. triggers a non regression test for issue #311
    rng = onp.random.RandomState(0)
    Q = rng.randn(7, 7)
    Q = onp.dot(Q, Q.T)
    A = rng.randn(4, 7)

    tmp = rng.randn(7)
    # Must have the same pytree structure as the output of matvec_Q.
    c = (tmp[:3], tmp[3:])
    # Must have the same pytree structure as the output of matvec_A.
    b = rng.randn(4)

    def matvec_Q(Q, tup):
      x_ = jnp.concatenate(tup)
      res = jnp.dot(Q, x_)
      return res[:3], res[3:]

    def matvec_A(A, tup):
      x_ = jnp.concatenate(tup)
      return jnp.dot(A, x_)

    # With pytrees directly.
    hyperparams = dict(params_obj=(Q, c), params_eq=(A, b))
    qp = EqualityConstrainedQP(matvec_Q=matvec_Q, matvec_A=matvec_A,
                               refine_regularization=refine_regularization)
    # sol.primal has the same pytree structure as the output of matvec_Q.
    # sol.dual_eq has the same pytree structure as the output of matvec_A.
    sol_pytree = qp.run(**hyperparams).params
    self.assertAllClose(
      qp.l2_optimality_error(sol_pytree, **hyperparams), 0.0, atol=1e-4
    )

    # With flattened pytrees.
    hyperparams = dict(params_obj=(Q, jnp.concatenate(c)), params_eq=(A, b))
    qp = EqualityConstrainedQP()
    sol = qp.run(**hyperparams).params
    self.assertAllClose(qp.l2_optimality_error(sol, **hyperparams), 0.0, atol=1e-4)

    # Check that the solutions match.
    self.assertArraysAllClose(jnp.concatenate(sol_pytree.primal), sol.primal, atol=1e-4)
    self.assertArraysAllClose(sol_pytree.dual_eq, sol.dual_eq, atol=1e-4)

  def test_fun_api(self):
    # mimic a function with pytree parameters.
    x = {'complicated_pytree': jnp.array([1.0, 2.0])}
    a = jnp.array([-0.5, 1.5])
    b = jnp.array(0.3)

    # convex quadratic and pytree parameters.
    def euclidean_quadratic_cost(y, params_obj):
      x = params_obj['complicated_pytree']
      return jnp.sum((x - y)*(x - y))
    
    def matvec_A(params_A, u):
      return jnp.dot(params_A, u).reshape(1)
    
    qp = EqualityConstrainedQP(fun=euclidean_quadratic_cost,
                               matvec_A=matvec_A)    

    try:
      # attempt to run without providing init_params, despite fun != None.
      hyperparams = dict(params_obj=x, params_eq=(a, b))
      sol = qp.run(**hyperparams).params
    except ValueError as e:
      pass  # here, a failure is the expected behavior.
    else:
      self.fail("Expected ValueError when init_params is not provided (`fun` API).")

    def scalar_solve_qp(hyperparams, init_params):
      params = qp.run(init_params, **hyperparams).params
      # arbitrary scalar function to ensure that the gradient is defined.
      return jnp.sum(params.primal)  

    hyperparams = dict(params_obj=x, params_eq=(a, jnp.array([b])))
    init_params = KKTSolution(jnp.array([1.0, 1.0]), jnp.array([1.0]))

    solve_value_grad = jax.value_and_grad(scalar_solve_qp)
    # we only need to test if the gradient is well defined in reverse mode AD:
    sol, unused_grad = solve_value_grad(hyperparams, init_params)

  def test_example_from_doc(self):
    """Test the example exposed in quadratic_problems.rst"""
    Q = jnp.array([[1.0, 0.5], [0.5, 4.0]])
    c = jnp.zeros(Q.shape[0])
    A = jnp.array([[1.0, 2.0]])
    b = jnp.array([1.0])

    def wrong_solver(Q):  # don't do this!

      def matvec_Q(params_Q, x):
        del params_Q  # unused
        # error! Q is captured from the global scope.
        # it does not fail now, but it will later.
        return jnp.dot(Q, x)
      
      eq_qp = EqualityConstrainedQP(matvec_Q=matvec_Q)
      # Warning: Q does not appear in the call to run()
      sol = eq_qp.run(None, params_obj=(None, c), params_eq=(A, b)).params
      loss = jnp.sum(sol.primal)
      return loss

    _ = wrong_solver(Q)  # no error... but it will fail later.
    try:
      _ = jax.grad(wrong_solver)(Q)  # error!
    except jax._src.interpreters.ad.CustomVJPException as e:
      pass  # here, a failure is the expected behavior.
    else:
      self.fail("Expected CustomVJPException when matvec_Q captures Q.")

    def correct_solver(Q):

      def matvec_Q(params_Q, x):
        return jnp.dot(params_Q, x)

      eq_qp = EqualityConstrainedQP(matvec_Q=matvec_Q)
      # Q is passed as a parameter, not captured from the global scope.
      sol = eq_qp.run(None, params_obj=(Q, c), params_eq=(A, b)).params
      loss = jnp.sum(sol.primal)
      return loss

    _ = correct_solver(Q)  # no error
    _ = jax.grad(correct_solver)(Q)  # no error

  def test_challenging_problem(self):
    size_prob = 100
    size_eq = 98

    onp.random.seed(511)
    X = onp.random.rand(size_prob, size_prob)
    Q = jnp.array(X @ X.T)
    c = jnp.array(onp.random.rand(size_prob))

    A = jnp.array(onp.random.rand(size_eq, size_prob))
    b = jnp.array(onp.random.rand(size_eq))

    # The tolerance is low, but ordinarily this problem cannot be solved by
    # EqualityConstrainedQP without refinement, because the matrix has a huge
    # condition number.
    low_tol = 1e-1
    qp = EqualityConstrainedQP(
      refine_regularization=1.0,
      refine_maxiter=3000,
      tol=low_tol,
      jit=True,
    )

    hyperparams = dict(params_obj=(Q, c), params_eq=(A, b))
    sol, state = qp.run(**hyperparams)
    error = qp.l2_optimality_error(sol, **hyperparams)
    self.assertLess(error, low_tol)


if __name__ == "__main__":
  absltest.main()
