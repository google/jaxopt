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
from jax import test_util as jtu
import jax.numpy as jnp

import jaxopt
from jaxopt import objective
from jaxopt import projection
from jaxopt import prox
from jaxopt._src import test_util

import optax

from sklearn import datasets


class CommonTest(jtu.JaxTestCase):

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
                   jaxopt.AndersonAcceleration(fixed_point_fun=fixed_point_fun),
                   jaxopt.AndersonWrapper(fpi),
                   jaxopt.ArmijoSGD(fun=fun)):

      hash(solver)  # Checks that hash works.
      update = jax.jit(solver.update)
      state0 = solver.init_state(params0)
      params, state = update(params0, state0, data=data)
      test_util.check_states_have_same_types(state0, state)

    # Proximal gradient solvers.
    for solver in (jaxopt.ProximalGradient(fun=fun, prox=prox.prox_lasso),
                   jaxopt.BlockCoordinateDescent(fun=fun,
                                                 block_prox=prox.prox_lasso)):

      hash(solver)  # Checks that hash works.
      update = jax.jit(solver.update)
      state0 = solver.init_state(params0, hyperparams_prox=1.0, data=data)
      params, state = update(params0, state0, hyperparams_prox=1.0, data=data)
      test_util.check_states_have_same_types(state0, state)

    # Projected gradient solvers.
    l2_proj = projection.projection_simplex

    for solver in (jaxopt.ProjectedGradient(fun=fun, projection=l2_proj),):

      hash(solver)  # Checks that hash works.
      update = jax.jit(solver.update)
      state0 = solver.init_state(params0, hyperparams_proj=1.0, data=data)
      params, state = update(params0, state0, hyperparams_proj=1.0, data=data)
      test_util.check_states_have_same_types(state0, state)

    # Bisection
    optimality_fun = lambda x: x ** 3 - x - 2
    bisec = jaxopt.Bisection(optimality_fun=optimality_fun, lower=1, upper=2)

    hash(bisec)  # Checks that hash works.
    update = jax.jit(bisec.update)
    state0 = bisec.init_state()
    params, state = update(params=None, state=state0)
    test_util.check_states_have_same_types(state0, state)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
