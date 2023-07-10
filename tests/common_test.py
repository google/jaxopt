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

from absl.testing import absltest

import jax
import jax.numpy as jnp

import jaxopt
from jaxopt import objective
from jaxopt import projection
from jaxopt import prox
from jaxopt._src import test_util

import optax

from sklearn import datasets


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

  def test_aux_consistency(self):
    def fun(w, X, y):
      residuals = jnp.dot(X, w) - y
      return jnp.sum(residuals ** 2), residuals

    X, y = datasets.make_classification(n_samples=10, n_features=5, n_classes=3,
                                        n_informative=3, random_state=0)


    for solver in (jaxopt.OptaxSolver(opt=optax.adam(1e-3), fun=fun,
                                      has_aux=True),
                   jaxopt.BFGS(fun=fun, has_aux=True),
                   jaxopt.LBFGS(fun=fun, has_aux=True),
                   jaxopt.NonlinearCG(fun=fun, has_aux=True),
                   jaxopt.GradientDescent(fun=fun, has_aux=True),
                   jaxopt.ArmijoSGD(fun=fun, has_aux=True),
                   jaxopt.PolyakSGD(fun=fun, has_aux=True),
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


if __name__ == '__main__':
  absltest.main()
