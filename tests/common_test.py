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

"""Common tests."""

from functools import partial
from absl.testing import absltest
from absl.testing import parameterized


from contextlib import redirect_stdout
import io


import jax
import jax.numpy as jnp
import numpy as onp
import optax
from sklearn import datasets
from sklearn import preprocessing

import jaxopt
from jaxopt import objective
from jaxopt import projection
from jaxopt import prox
from jaxopt import tree_util as tu
from jaxopt._src import test_util
from jaxopt._src.base import LinearOperator
from jaxopt._src.objective import LeastSquares, MulticlassLinearSvmDual
from jaxopt._src.osqp import BoxOSQP
from jaxopt._src.osqp import OSQP


N_CALLS = 0

def check_states_have_same_types(state1, state2):
  if len(state1._fields) != len(state2._fields):
    raise ValueError("state1 and state2 should have the same number of "
                     "attributes.")

  for attr1, attr2 in zip(state1._fields, state2._fields):
    if attr1 != attr2:
      raise ValueError("Attribute names do not agree: %s and %s." % (attr1,
                                                                     attr2))

    type1 = type(getattr(state1, attr1)).__name__
    type2 = type(getattr(state2, attr2)).__name__

    if type1 != type2:
      raise ValueError("Attribute '%s' has different types in state1 and "
                       "state2: %s vs. %s" % (attr1, type1, type2))


class CommonTest(test_util.JaxoptTestCase):

  def test_jit_update_and_returned_states(self):
    fun = objective.least_squares
    root_fun = jax.grad(fun)

    def fixed_point_fun(params, *args, **kwargs):
      return root_fun(params, *args, **kwargs) - params


    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)
    data = (X, y)
    params0 = jnp.zeros(X.shape[1])

    fpi = jaxopt.FixedPointIteration(fixed_point_fun=fixed_point_fun)

    # Gradient solvers and fixed point
    for solver in (jaxopt.GradientDescent(fun=fun),
                   jaxopt.PolyakSGD(fun=fun),
                   jaxopt.OptaxSolver(opt=optax.adam(1e-3), fun=fun),
                   fpi,
                   jaxopt.Broyden(fun=root_fun),
                   jaxopt.AndersonAcceleration(fixed_point_fun=fixed_point_fun),
                   jaxopt.AndersonWrapper(fpi),
                   jaxopt.ArmijoSGD(fun=fun),
                   jaxopt.BFGS(fun),
                   jaxopt.LBFGS(fun=fun)):

      hash(solver)  # Checks that hash works.
      update = jax.jit(solver.update)
      state0 = solver.init_state(params0, data=data)
      params, state = update(params0, state0, data=data)
      check_states_have_same_types(state0, state)

    # Proximal gradient solvers.
    for solver in (jaxopt.ProximalGradient(fun=fun, prox=prox.prox_lasso),
                   jaxopt.BlockCoordinateDescent(fun=fun,
                                                 block_prox=prox.prox_lasso)):

      hash(solver)  # Checks that hash works.
      update = jax.jit(solver.update)
      state0 = solver.init_state(params0, hyperparams_prox=1.0, data=data)
      params, state = update(params0, state0, hyperparams_prox=1.0, data=data)
      check_states_have_same_types(state0, state)

    # Projected gradient solvers.
    l2_proj = projection.projection_simplex

    for solver in (jaxopt.ProjectedGradient(fun=fun, projection=l2_proj),):

      hash(solver)  # Checks that hash works.
      update = jax.jit(solver.update)
      state0 = solver.init_state(params0, hyperparams_proj=1.0, data=data)
      params, state = update(params0, state0, hyperparams_proj=1.0, data=data)
      check_states_have_same_types(state0, state)

    # Bisection
    optimality_fun = lambda x: x ** 3 - x - 2
    bisec = jaxopt.Bisection(optimality_fun=optimality_fun, lower=1, upper=2)

    hash(bisec)  # Checks that hash works.
    update = jax.jit(bisec.update)
    state0 = bisec.init_state()
    params, state = update(params=None, state=state0)
    check_states_have_same_types(state0, state)

  def test_n_calls(self):
    global N_CALLS

    class LeastSquaresNCalls(LeastSquares):
      def __init__(self, with_custom_linop=False):
        super().__init__()
        self.with_custom_linop = with_custom_linop

      def __call__(self, W, data):
        global N_CALLS
        N_CALLS += 1
        return super().__call__(W, data)

      def make_linop(self, data):
        """Creates linear operator."""
        if self.with_custom_linop:
          return LinopNCall(data[0])
        else:
          return super().make_linop(data)

    class LinopNCall(LinearOperator):
      def matvec(self, x):
        """Computes dot(A, x)."""
        global N_CALLS
        N_CALLS += 1
        return jnp.dot(self.A, x)

      def rmatvec_element(self, x, idx):
        """Computes dot(A.T, x)[idx]."""
        global N_CALLS
        N_CALLS += 1
        return jnp.dot(self.A[:, idx], x)

    fun = LeastSquaresNCalls()
    root_fun = jax.grad(fun)

    def fixed_point_fun(params, *args, **kwargs):
      return root_fun(params, *args, **kwargs) - params


    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)
    data = (X, y)
    params0 = jnp.zeros(X.shape[1])
    common_kwargs = dict(jit=False, maxiter=30)
    fpi = jaxopt.FixedPointIteration(fixed_point_fun=fixed_point_fun, **common_kwargs)

    # Gradient solvers and fixed point
    for solver in (jaxopt.PolyakSGD(fun=fun, **common_kwargs),
                   fpi,
                   jaxopt.Broyden(fun=root_fun, **common_kwargs),
                   jaxopt.AndersonAcceleration(fixed_point_fun=fixed_point_fun, **common_kwargs),
                   jaxopt.NonlinearCG(fun, **common_kwargs),
                   jaxopt.BFGS(fun, **common_kwargs),
                   jaxopt.LBFGS(fun=fun, **common_kwargs)):

      _, state = solver.run(params0, data=data)
      self.assertEqual(state.num_fun_eval, N_CALLS)
      N_CALLS = 0

    # # Proximal gradient solvers.
    # FIXME: currently the following test does not work because
    # of a fori_loop issue in block cd (does not allow non-jitted functions). (zramzi)
    # fun = LeastSquaresNCalls(with_custom_linop=True)
    # for solver in (jaxopt.BlockCoordinateDescent(fun=fun,
    #                                              block_prox=prox.prox_lasso,
    #                                              **common_kwargs),):

    #   _, state = solver.run(params0, hyperparams_prox=1.0, data=data)
    #   self.assertEqual(state.num_fun_eval, N_CALLS)
    #   N_CALLS = 0

    # Mirror Descent
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    Y = jnp.asarray(Y)
    n_samples, n_classes = Y.shape

    class MulticlassLinearSvmDualNCall(MulticlassLinearSvmDual):
      def __call__(self, params, *args, **kwargs):
        global N_CALLS
        N_CALLS += 1
        return super().__call__(params, *args, **kwargs)

    fun = MulticlassLinearSvmDualNCall()
    lam = 10.0
    data = (X, Y)

    beta_init = jnp.ones((n_samples, n_classes)) / n_classes
    def kl_projection(x, hyperparams_proj):
      del hyperparams_proj
      return tu.tree_map(lambda t: jax.nn.softmax(t, -1), x)
    # Generating function of the Bregman divergence.
    kl_generating_fun = lambda x: -jnp.sum(jax.scipy.special.entr(x) + x)
    # Row-wise mirror map.
    kl_mapping_fun = jax.vmap(jax.grad(kl_generating_fun))
    projection_grad = jaxopt.MirrorDescent.make_projection_grad(
          kl_projection, kl_mapping_fun)
    md = jaxopt.MirrorDescent(
        fun=fun,
        projection_grad=projection_grad,
        stepsize=1e-3,
        **common_kwargs)
    _, state = md.run(beta_init, None, lam, data)
    self.assertEqual(state.num_fun_eval, N_CALLS)
    N_CALLS = 0

    # Bisection
    def opt_fun(x):
      global N_CALLS
      N_CALLS += 1
      return x ** 3 - x - 2

    bisec = jaxopt.Bisection(
      optimality_fun=opt_fun,
      lower=1,
      upper=2,
      **common_kwargs,
    )

    _, state = bisec.run()
    self.assertEqual(state.num_fun_eval, N_CALLS)
    N_CALLS = 0

  @parameterized.product(value_and_grad=[True, False])
  def test_aux_consistency(self, value_and_grad):
    def fun_(w, X, y):
      residuals = jnp.dot(X, w) - y
      return jnp.sum(residuals ** 2), residuals

    if value_and_grad:
      fun = jax.value_and_grad(fun_, has_aux=True)
    else:
      fun = fun_
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)

    for solver in (jaxopt.OptaxSolver(opt=optax.adam(1e-3), fun=fun, has_aux=True,
                                      value_and_grad=value_and_grad),
                   jaxopt.BFGS(fun=fun, has_aux=True,
                               value_and_grad=value_and_grad),
                   jaxopt.LBFGS(fun=fun, has_aux=True,
                                value_and_grad=value_and_grad),
                   jaxopt.NonlinearCG(fun=fun, has_aux=True,
                                      value_and_grad=value_and_grad),
                   jaxopt.GradientDescent(fun=fun, has_aux=True,
                                          value_and_grad=value_and_grad),
                   jaxopt.ArmijoSGD(fun=fun, has_aux=True,
                                    value_and_grad=value_and_grad),
                   jaxopt.PolyakSGD(fun=fun, has_aux=True,
                                    value_and_grad=value_and_grad),
    ):

      params = jnp.zeros(X.shape[1])
      state1 = solver.init_state(params, X, y)
      params, state2 = solver.update(params, state1, X, y)
      self.assertEqual(type(state1.aux), type(state2.aux))

  def test_dtype_consistency(self):
    # We don't use float32 (default dtype in JAX)
    # to detect when the dtype is incorrectly inherited.
    # We use two different dtypes for function outputs and
    # for parameters, as it's a common use case.
    dtype_fun = jnp.bfloat16
    dtype_params = jnp.float16

    def fun(w, X, y):
      w = w.astype(dtype_fun)
      return jnp.sum((jnp.dot(X, w) - y) ** 2)

    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)
    X = jnp.array(X, dtype=dtype_fun)
    y = jnp.array(y, dtype=dtype_fun)
    init = jnp.zeros(X.shape[1], dtype=dtype_params)

    for solver in (
                   jaxopt.ArmijoSGD(fun=fun),
                   jaxopt.BFGS(fun, linesearch="zoom"),
                   jaxopt.BFGS(fun, linesearch="backtracking"),
                   jaxopt.GradientDescent(fun=fun, acceleration=True),
                   jaxopt.GradientDescent(fun=fun, acceleration=False),
                   jaxopt.GradientDescent(fun=fun, stepsize=1e-3),
                   jaxopt.LBFGS(fun=fun, linesearch="zoom"),
                   jaxopt.LBFGS(fun=fun, linesearch="backtracking"),
                   jaxopt.NonlinearCG(fun, linesearch="zoom"),
                   jaxopt.NonlinearCG(fun, linesearch="backtracking"),
                   jaxopt.OptaxSolver(opt=optax.adam(1e-3), fun=fun),
                   jaxopt.PolyakSGD(fun=fun, momentum=0.0),
                   jaxopt.PolyakSGD(fun=fun, momentum=0.9),
                   jaxopt.ProjectedGradient(fun=fun,
                     projection=jaxopt.projection.projection_non_negative),
    ):

      state = solver.init_state(init, X=X, y=y)
      params, state = solver.update(init, state, X=X, y=y)

      self.assertEqual(params.dtype, dtype_params)

      solver_name = solver.__class__.__name__

      if hasattr(state, "value"):
        # FIXME: ideally, solver states should always include a value attribute.
        self.assertEqual(state.value.dtype, dtype_fun,
                         msg="value dtype error in solver '%s'" % solver_name)

      # Usually, the error is computed from the gradient,
      # which has the same dtype as the parameters.
      self.assertEqual(state.error.dtype, dtype_params)

  def test_weak_type_consistency(self):
    def fun(w, X, y):
      return jnp.sum((jnp.dot(X, w) - y) ** 2)

    solvers = (
       jaxopt.ArmijoSGD(fun=fun),
       jaxopt.BFGS(fun, linesearch="zoom"),
       jaxopt.BFGS(fun, linesearch="backtracking"),
       jaxopt.GradientDescent(fun=fun, acceleration=True),
       jaxopt.GradientDescent(fun=fun, acceleration=False),
       jaxopt.LBFGS(fun=fun, linesearch="zoom"),
       jaxopt.LBFGS(fun=fun, linesearch="backtracking"),
       jaxopt.NonlinearCG(fun, linesearch="zoom"),
       jaxopt.NonlinearCG(fun, linesearch="backtracking"),
       jaxopt.OptaxSolver(opt=optax.adam(1e-3), fun=fun),
       jaxopt.PolyakSGD(fun=fun, momentum=0.0),
       jaxopt.PolyakSGD(fun=fun, momentum=0.9),
       jaxopt.ProjectedGradient(fun=fun,
         projection=jaxopt.projection.projection_non_negative),
    )

    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)
    X = jnp.array(X)
    y = jnp.array(y)
    init = jnp.zeros(X.shape[1])

    for solver in solvers:
      state0 = solver.init_state(init, X=X, y=y)
      params, state1 = solver.update(init, state0, X=X, y=y)
      for field in state0._fields:
        field0 = getattr(state0, field)
        field1 = getattr(state1, field)

        weak_type0 = getattr(field0, "weak_type", None)
        weak_type1 = getattr(field1, "weak_type", None)

        solver_name = solver.__class__.__name__
        msg = "weak_type inconsistency for attribute '%s' in solver '%s'"
        self.assertEqual(weak_type0, weak_type1, msg=msg % (field, solver_name))

  @parameterized.product(verbose=[True, False])
  def test_jit_with_or_without_verbose(self, verbose):

    fun = lambda p: p @ p

    def fixed_point_fun(params):
      return fun(params) - params

    solvers = (
      # Unconstrained
      jaxopt.GradientDescent(fun=fun, jit=True, verbose=verbose, maxiter=4),
      jaxopt.PolyakSGD(fun=fun, jit=True, verbose=verbose, maxiter=4),
      jaxopt.Broyden(fun=fixed_point_fun, jit=True, verbose=verbose, maxiter=4),
      jaxopt.AndersonAcceleration(fixed_point_fun=fixed_point_fun, jit=True, verbose=verbose, maxiter=4),
      jaxopt.ArmijoSGD(fun=fun, jit=True, verbose=verbose, maxiter=4),
      jaxopt.BFGS(fun, linesearch="zoom", jit=True, verbose=verbose, maxiter=4),
      jaxopt.BFGS(fun, linesearch="backtracking", jit=True, verbose=verbose, maxiter=4),
      jaxopt.BFGS(fun, linesearch="hager-zhang", jit=True, verbose=verbose, maxiter=4),
      jaxopt.LBFGS(fun=fun, jit=True, verbose=verbose, maxiter=4),
      jaxopt.ArmijoSGD(fun=fun, jit=True, verbose=verbose, maxiter=4),
      jaxopt.NonlinearCG(fun, jit=True, verbose=verbose, maxiter=4),
      # Unconstrained, nonlinear least-squares
      jaxopt.GaussNewton(residual_fun=fun, jit=True, verbose=verbose, maxiter=4),
      jaxopt.LevenbergMarquardt(residual_fun=fun, jit=True, verbose=verbose, maxiter=4),
      # Constrained
      jaxopt.ProjectedGradient(fun=fun,
        projection=jaxopt.projection.projection_non_negative, jit=True, verbose=verbose, maxiter=4),
      # Optax wrapper
      jaxopt.OptaxSolver(opt=optax.adam(1e-1), fun=fun, jit=True, verbose=verbose, maxiter=4),
    )

    @partial(jax.jit, static_argnums=(1,))
    def run_solver(p0, solver):
        return solver.run(p0)

    for solver in solvers:
      stdout = io.StringIO()
      with redirect_stdout(stdout):
        jax.block_until_ready(run_solver(jnp.arange(2.), solver))
      printed = len(stdout.getvalue()) > 0
      if verbose:
        self.assertTrue(printed)
      else:
        self.assertFalse(printed)

    # Proximal gradient solvers
    fun = objective.least_squares
    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)
    data = (X, y)
    params0 = jnp.zeros(X.shape[1])

    @partial(jax.jit, static_argnums=(1,))
    def run_solver_prox(p0, solver):
      return solver.run(p0, hyperparams_prox=1.0, data=data)

    for solver in (jaxopt.ProximalGradient(fun=fun, prox=prox.prox_lasso,
                                           jit=True, verbose=verbose, maxiter=4),
                   jaxopt.BlockCoordinateDescent(fun=fun,
                                                 block_prox=prox.prox_lasso,
                                                 jit=True, verbose=verbose, maxiter=4)
    ):
      stdout = io.StringIO()
      with redirect_stdout(stdout):
        jax.block_until_ready(run_solver_prox(params0, solver))
      printed = len(stdout.getvalue()) > 0
      if verbose:
        self.assertTrue(printed)
      else:
        self.assertFalse(printed)

    # Mirror Descent
    Y = preprocessing.LabelBinarizer().fit_transform(y)
    Y = jnp.asarray(Y)
    n_samples, n_classes = Y.shape

    fun = MulticlassLinearSvmDual()
    lam = 10.0
    data = (X, Y)

    beta_init = jnp.ones((n_samples, n_classes)) / n_classes
    def kl_projection(x, hyperparams_proj):
      del hyperparams_proj
      return tu.tree_map(lambda t: jax.nn.softmax(t, -1), x)
    # Generating function of the Bregman divergence.
    kl_generating_fun = lambda x: -jnp.sum(jax.scipy.special.entr(x) + x)
    # Row-wise mirror map.
    kl_mapping_fun = jax.vmap(jax.grad(kl_generating_fun))
    projection_grad = jaxopt.MirrorDescent.make_projection_grad(
          kl_projection, kl_mapping_fun)

    @jax.jit
    def run_mirror_descent(b0):
      md = jaxopt.MirrorDescent(
          fun=fun,
          projection_grad=projection_grad,
          stepsize=1e-3,
          maxiter=4,
          jit=True,
          verbose=verbose)
      _, state = md.run(b0, None, lam, data)
      return state

    stdout = io.StringIO()
    with redirect_stdout(stdout):
      jax.block_until_ready(run_mirror_descent(beta_init))
    printed = len(stdout.getvalue()) > 0
    if verbose:
      self.assertTrue(printed)
    else:
      self.assertFalse(printed)

    # Quadratic programming - BoxOSQP
    x = jnp.array([1.0, 2.0])
    a = jnp.array([-0.5, 1.5])
    b = 0.3
    q = -x
    # Find ||y-x||^2 such that jnp.dot(y, a) = b.

    matvec_Q = lambda params_Q,u: u
    matvec_A = lambda params_A,u: jnp.dot(a, u).reshape(1)

    tol = 1e-4

    @jax.jit
    def run_box_osqp(params_obj, params_ineq):
      osqp = BoxOSQP(matvec_Q=matvec_Q, matvec_A=matvec_A, tol=tol, jit=True, verbose=verbose, maxiter=4)
      return osqp.run(None, (None, params_obj), None, (params_ineq, params_ineq))

    stdout = io.StringIO()
    with redirect_stdout(stdout):
      jax.block_until_ready(run_box_osqp(q, b))
    printed = len(stdout.getvalue()) > 0
    if verbose:
      self.assertTrue(printed)
    else:
      self.assertFalse(printed)


if __name__ == '__main__':
  absltest.main()
