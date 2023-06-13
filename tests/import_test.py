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

import jaxopt
from jaxopt._src import test_util


class ImportTest(test_util.JaxoptTestCase):

  def test_implicit_diff(self):
    jaxopt.implicit_diff.root_vjp
    from jaxopt.implicit_diff import root_vjp

  def test_isotonic(self):
    jaxopt.isotonic.isotonic_l2_pav
    from jaxopt.isotonic import isotonic_l2_pav

  def test_prox(self):
    jaxopt.prox.prox_none
    from jaxopt.prox import prox_none

  def test_projection(self):
    jaxopt.projection.projection_simplex
    from jaxopt.projection import projection_simplex

  def test_tree_util(self):
    from jaxopt.tree_util import tree_vdot

  def test_linear_solve(self):
    from jaxopt.linear_solve import solve_lu

  def test_base(self):
    from jaxopt.base import LinearOperator

  def test_perturbations(self):
    from jaxopt.perturbations import make_perturbed_argmax

  def test_loss(self):
    jaxopt.loss.binary_logistic_loss
    from jaxopt.loss import binary_logistic_loss

  def test_objective(self):
    jaxopt.objective.least_squares
    from jaxopt.objective import least_squares

  def test_loop(self):
    from jaxopt.loop import while_loop


if __name__ == '__main__':
  absltest.main()
