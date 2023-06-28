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

from jaxopt import implicit_diff
from jaxopt import isotonic
from jaxopt import loss
from jaxopt import objective
from jaxopt import projection
from jaxopt import prox

from jaxopt._src.anderson import AndersonAcceleration
from jaxopt._src.anderson_wrapper import AndersonWrapper
from jaxopt._src.armijo_sgd import ArmijoSGD
from jaxopt._src.base import OptStep
from jaxopt._src.backtracking_linesearch import BacktrackingLineSearch
from jaxopt._src.bfgs import BFGS
from jaxopt._src.bisection import Bisection
from jaxopt._src.block_cd import BlockCoordinateDescent
from jaxopt._src.broyden import Broyden
from jaxopt._src.cd_qp import BoxCDQP
from jaxopt._src.cvxpy_wrapper import CvxpyQP
from jaxopt._src.eq_qp import EqualityConstrainedQP
from jaxopt._src.fixed_point_iteration import FixedPointIteration
from jaxopt._src.gauss_newton import GaussNewton
from jaxopt._src.gradient_descent import GradientDescent
from jaxopt._src.hager_zhang_linesearch import HagerZhangLineSearch
from jaxopt._src.iterative_refinement import IterativeRefinement
from jaxopt._src.lbfgs import LBFGS
from jaxopt._src.lbfgsb import LBFGSB
from jaxopt._src.levenberg_marquardt import LevenbergMarquardt
from jaxopt._src.mirror_descent import MirrorDescent
from jaxopt._src.nonlinear_cg import NonlinearCG
from jaxopt._src.optax_wrapper import OptaxSolver
from jaxopt._src.osqp import BoxOSQP
from jaxopt._src.osqp import OSQP
from jaxopt._src.polyak_sgd import PolyakSGD
from jaxopt._src.projected_gradient import ProjectedGradient
from jaxopt._src.proximal_gradient import ProximalGradient
from jaxopt._src.scipy_wrappers import ScipyBoundedLeastSquares
from jaxopt._src.scipy_wrappers import ScipyBoundedMinimize
from jaxopt._src.scipy_wrappers import ScipyLeastSquares
from jaxopt._src.scipy_wrappers import ScipyMinimize
from jaxopt._src.scipy_wrappers import ScipyRootFinding
from jaxopt._src.zoom_linesearch import ZoomLineSearch
