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

"""Preconditioned solvers for equality constrained quadratic programming."""

from typing import Optional, Any
from dataclasses import dataclass
import jax.numpy as jnp
import jaxopt
from jaxopt._src import base
from jaxopt._src import linear_operator


@dataclass
class PseudoInversePreconditionedEqQP(base.Solver):
  qp_solver: jaxopt.EqualityConstrainedQP

  def init_params(self, params_obj, params_eq):
    """Computes the matvec associated to the pseudo inverse of the KKT matrix."""
    Q, p = params_obj
    A, b = params_eq
    del p, b

    kkt_mat = jnp.block([[Q, A.T], [A, jnp.zeros((A.shape[0], A.shape[0]))]])
    kkt_mat_pinv = jnp.linalg.pinv(kkt_mat)

    d = Q.shape[0]

    pinv_blocks = (
      (kkt_mat_pinv[:d, :d], kkt_mat_pinv[:d, d:]),
      (kkt_mat_pinv[d:, :d], kkt_mat_pinv[d:, d:]),
    )
    return linear_operator.BlockLinearOperator(pinv_blocks)

  def run(
    self,
    init_params: Optional[base.KKTSolution] = None,
    params_obj: Optional[Any] = None,
    params_eq: Optional[Any] = None,
    params_precond=None,
    **kwargs
  ):
    # TODO(gnegiar): the M parameter should be passed to both
    # the QP solve and the implicit_diff_solve
    return self.qp_solver.run(
      init_params, params_obj, params_eq, M=params_precond, **kwargs
    )
