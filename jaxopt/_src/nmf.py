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

"""Non Negative Matrix Factorization."""

from dataclasses import dataclass
from optparse import Option
from typing import NamedTuple, Optional, Callable, Tuple, Union

import jax
import jax.numpy as jnp

from jaxopt import base 
from jaxopt._src import implicit_diff as idf
import jaxopt.base as base

from sklearn.decomposition import non_negative_factorization
from sklearn.decomposition._nmf import _initialize_nmf as sklearn_initialize_nmf
from sklearn.decomposition import NMF as sk_NMF


def frobenius_norm_squared(vec):
  return jnp.sum(vec**2)

def _make_nnls_optimality_fun():
  def obj_fun(primal_var, params_obj):
    H = primal_var
    Y, W = params_obj
    return 0.5 * frobenius_norm_squared(Y - W @ H.T)

  def ineq_fun(primal_var, params_ineq):
    H = primal_var
    return -H  # H >= 0  <=>  -H <= 0

  return idf.make_kkt_optimality_fun(obj_fun=obj_fun, eq_fun=None, ineq_fun=ineq_fun)


class NNLSState(NamedTuple):
  """Named tuple containing state information.
  Attributes:
    iter_num: iteration number.
    error: error used as stop criterion, deduced from residuals.
    primal_residuals: relative residuals primal problem.
    dual_residuals: relative residuals dual problem.
    rho: step size in ADMM.
    H_bar: previous value of H_bar, useful for warm start.  
  """
  iter_num: int
  error: float
  primal_residuals: jnp.array
  dual_residuals: jnp.array
  rho: float
  H_bar: jnp.array

@dataclass(eq=False)
class NNLS(base.IterativeSolver):
  """ Non Negative Least Squares solver based on ADMM.

  Solves ::
    min_H 0.5 * ||Y - W @ H.T||_F^2
    s.t.  H >= 0

  Based on ADMM algorithm [2] for matrix factorization [1].

  Args:
    maxiter: maximum number of iterations.
    tol: tolerance for stopping criterion.
    cg_tol: tolerance of inner conjugate gradient solver.
    verbose: If verbose=1, print error at each iteration.
      Warning: verbose>0 will automatically disable jit.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").
    eps: epsilon for safe division in evaluation of residuals.
    rho_policy: policy for rho updates.
      float: constant step size. Often requires tuning for fast convergence.
      'osqp': for OSQP-like policy from paper [3].
      'adaptive': strategy from the paper [1].
    rho_min: minimum value of rho.
    rho_max: maximum value of rho.

  References:

  [1] Huang, K., Sidiropoulos, N.D. and Liavas, A.P., 2016.
      A flexible and efficient algorithmic framework for constrained matrix and tensor factorization.
      IEEE Transactions on Signal Processing, 64(19), pp.5052-5065.

  [2] Boyd, S., Parikh, N., Chu, E., Peleato, B. and Eckstein, J., 2010.
      Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers.
      Machine Learning, 3(1), pp.1-122.

  [3] Stellato, B., Banjac, G., Goulart, P., Bemporad, A. and Boyd, S., 2020.
      OSQP: An operator splitting solver for quadratic programs.
      Mathematical Programming Computation, 12(4), pp.637-672.
  """
  maxiter: int = 1000
  tol: float = 1e-4
  cg_tol: float = 1e-6
  verbose: int = 0
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"
  eps: float = 1e-8
  rho_policy: Union[float, str] = 'osqp'
  rho_min: float = 1e-6
  rho_max: float = 1e6

  def init_params(self,
                  params_obj: Tuple[jnp.array, jnp.array] = None,
                  params_eq: None = None,
                  params_ineq: None = None):
    Y, W = params_obj
    m, k = Y.shape[1], W.shape[1]
    U = jnp.zeros((m, k), dtype=Y.dtype)
    H = U
    kkt_sol = base.KKTSolution(primal=H,
                               dual_eq=None,
                               dual_ineq=U)
    return kkt_sol

  def compute_residuals(self, H_previous, H, H_bar, U):
    # Residuals proposed by [1].
    # Different from those of OSQP: faster to compute.
    primal_residuals = frobenius_norm_squared(H - H_bar.T) / (frobenius_norm_squared(H) + self.eps)
    dual_residuals = frobenius_norm_squared(H - H_previous) / (frobenius_norm_squared(U) + self.eps)
    return primal_residuals, dual_residuals

  def _compute_rho(self, Y, W, H, H_bar, U, rho):
    if not isinstance(self.rho_policy, str):
      # constant value: rho = rho_policy
      return self.rho_policy
    
    if self.rho_policy == 'adaptive':
      # Adaptive policy from original's paper [1]
      # Sometimes fails in some situations for implicit diff.
      k = W.shape[1]
      rho_adaptive = jnp.asarray(frobenius_norm_squared(W) / k)
      return jnp.clip(rho_adaptive, self.rho_min, self.rho_max)

    if self.rho_policy == 'osqp':
      # OSQP-like policy (since NNLS is a quadratic program)
      # Freely inspired by OSQP's paper.
      osqp_Px = W.T @ W @ H_bar
      osqp_q = -W.T @ Y
      inf_norm = lambda M: jnp.max(jnp.abs(M))  # infty norm for matrices seen as vectors
      osqp_primal = inf_norm(H - H_bar.T) + self.eps
      osqp_dual = inf_norm(osqp_Px + osqp_q + U.T) + self.eps
      a = osqp_primal / (jnp.maximum(inf_norm(H_bar), inf_norm(H)) + self.eps)
      b = osqp_dual / (jnp.maximum(jnp.maximum(inf_norm(osqp_Px), inf_norm(osqp_q)), inf_norm(U.T)) + self.eps)
      rho_osqp = ((a / b)**0.5) * rho
      return jnp.clip(rho_osqp, self.rho_min, self.rho_max)

    raise ValueError(f"Unrecognized option {self.rho_policy} for NNLS.rho_policy")

  def init_state(self, init_params,
                 params_obj: Tuple[jnp.array, jnp.array],
                 params_eq: None = None,
                 params_ineq: None = None):
    Y, W = params_obj
    H = init_params.primal
    U = init_params.dual_ineq
    H_bar = init_params.primal.T
    rho = self._compute_rho(Y, W, H, H_bar, U, 0.1)
    state = NNLSState(
        iter_num=jnp.asarray(0, dtype=jnp.int32),
        error=jnp.asarray(jnp.inf),
        primal_residuals=jnp.asarray(jnp.inf),
        dual_residuals=jnp.asarray(jnp.inf),
        rho=rho,
        H_bar=H_bar,
    )
    return state

  def _compute_H_bar(self, H, G, F, U, rho, H_bar):
    # solution to argmin_H \|H - H_bar + U\|_2 + r(H)
    # G is PSD so G + rho * I is PSD => Conjugate Gradient can be used.
    def matmul(vec):
      return jnp.dot(G, vec) + rho * vec
    right_member = F + rho * (H + U).T
    H_bar, _ = jax.scipy.sparse.linalg.cg(matmul, right_member,
                                          x0=H_bar, tol=self.cg_tol)
    return H_bar

  def update(self, params, state, params_obj, params_eq, params_ineq):
    """Update state of the NNLS.

    n: number of rows
    m: number of columns
    k: rank of low rank factorization

    Args:
      params: KKTSolution tuple, with params.primal = H and H of shape (m, k)
      state: NNLSState object.
      params_obj: pair (Y, W), Y of shape (n, m) and W of shape (n, k)
      params_eq: None, present for signature purposes.
      params_ineq: None, present for signature purposes.

    Returns:
      pair params, 
    """
    Y, W = params_obj
    F = W.T @ Y
    G = W.T @ W  # PSD matrix.
    H, U = params.primal, params.dual_ineq

    # ADMM first inner problem.
    H_bar = self._compute_H_bar(H, G, F, U, state.rho, state.H_bar)

    # ADMM second inner problem.
    H = jax.nn.relu(H_bar.T - U)

    # Gradient ascent step on dual variables.
    U = U + (H - H_bar.T)

    primal_residuals, dual_residuals = self.compute_residuals(params.primal, H, H_bar, U)
    error = jnp.maximum(primal_residuals, dual_residuals)

    rho = self._compute_rho(Y, W, H, H_bar, U, state.rho)

    next_params = base.KKTSolution(primal=H, dual_eq=None, dual_ineq=U)

    next_state = NNLSState(
        iter_num=state.iter_num+1,
        error=error,
        primal_residuals=primal_residuals,
        dual_residuals=dual_residuals,
        rho=rho,
        H_bar=H_bar,
    )
    return base.OptStep(next_params, next_state)

  def run(self,
          init_params: Optional[base.KKTSolution],
          params_obj: Tuple[jnp.array, jnp.array],
          params_eq: None = None,
          params_ineq: None = None):
    """Run the NNLS algorithm.
    
    Args:
      init_params: (optional) KKTSolution tuple, with params.primal = H and H of shape (m, k).
        When None, the algorithm will use `init` to initialize H.
        Only 'zeros' initialization is supported when `jit=True`.
      params_obj: pair (Y, W), Y of shape (n, m) and W of shape (n, k).
      params_eq: None, present for signature purposes.
      params_ineq: None, present for signature purposes.
    """
    if init_params is None:
      init_params = self.init_params(params_obj, params_eq, params_ineq)
    return super().run(init_params, params_obj, params_eq, params_ineq)

  def __post_init__(self):
    self.optimality_fun = _make_nnls_optimality_fun()


def _make_nmf_kkt_optimality_fun():
  def obj_fun(primal_var, params_obj):
    H1, H2 = primal_var
    Y      = params_obj
    return 0.5 * frobenius_norm_squared(Y - H1 @ H2.T)

  def ineq_fun(primal_var, params_ineq):
    H1, H2 = primal_var
    return -H1, -H2 # -H1 <= 0 and -H2 <= 0

  return idf.make_kkt_optimality_fun(obj_fun=obj_fun, eq_fun=None, ineq_fun=ineq_fun)


class NMFState(NamedTuple):
  """Named tuple containing state information.

  Attributes:
    iter_num: iteration number.
    error: error used as stop criterion, deduced from residuals. 
    nnls_state_1: NNLState associated to H1.
    nnls_state_2: NNLState associated to H2.
  """
  iter_num: int
  error: float
  nnls_state_1: NNLSState
  nnls_state_2: NNLSState


@dataclass(eq=False)
class NMF(base.IterativeSolver):
  """ Non Negative Matrix Factorization solver based on ADMM.

  Solves ::
    min_{H1, H2} 0.5 * ||Y - H1 @ H2.T||_F^2
    s.t.  H1 >= 0, H2 >= 0

  Based on ADMM algorithm [2] for matrix factorization [1].

  This problem is NP-hard [3] and Non Convex in general.
  However, it is bi-convex: convex wrt to H1 (resp. H2) when H2 (resp. H1) is held constant.  

  Hence, it is possible to solve it by sequentially solving the two convex subproblems.
  It yields a solution that is a good approximation to the optimal solution.
  H1 (resp. H2) is differentiable wrt to Y when H2 (resp. H1) is held constant. 
  Nonetheless, those two derivatives can be used as an approximation of the gradient of the joint solution (H1, H2).

  Args:
    rank: rank of factors W, H.
    maxiter: maximum number of iterations.
    tol: tolerance for stopping criterion.
    cg_tol: tolerance of inner conjugate gradient solver.
    init: initialization of H.
      'sklearn_*': use sklearn's initialization schemes, where '*' is an initialization scheme of sklearn.
    verbose: If verbose=1, print error at each iteration.
      Warning: verbose>0 will automatically disable jit.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").
    nnls_solver: solver to solve inner NNLS problem. Defaults to NNLS().

  References:

  [1] Huang, K., Sidiropoulos, N.D. and Liavas, A.P., 2016.
      A flexible and efficient algorithmic framework for constrained matrix and tensor factorization.
      IEEE Transactions on Signal Processing, 64(19), pp.5052-5065.

  [2] Boyd, S., Parikh, N., Chu, E., Peleato, B. and Eckstein, J., 2010.
      Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers.
      Machine Learning, 3(1), pp.1-122.

  [3] Vavasis, S.A., 2010.
      On the complexity of nonnegative matrix factorization.
      SIAM Journal on Optimization, 20(3), pp.1364-1377.
  """
  rank: int
  maxiter: int = 1000
  tol: float = 1e-3
  init: str = 'sklearn_nndsvda'
  verbose: int = 0
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"
  nnls_solver: base.IterativeSolver = NNLS()

  def _split_kkt_sol(self, params):
    H1, H2       = params.primal
    U1, U2       = params.dual_ineq
    kkt_sol_1    = base.KKTSolution(primal=H1, dual_eq=None, dual_ineq=U1)
    kkt_sol_2    = base.KKTSolution(primal=H2, dual_eq=None, dual_ineq=U2)
    return kkt_sol_1, kkt_sol_2

  def _merge_kkt_sol(self, kkt_sol_1, kkt_sol_2):
    primal = kkt_sol_1.primal, kkt_sol_2.primal
    dual_ineq = kkt_sol_1.dual_ineq, kkt_sol_2.dual_ineq
    return base.KKTSolution(primal=primal, dual_eq=None, dual_ineq=dual_ineq)

  def init_params(self,
                  params_obj: jnp.array,
                  params_eq: None = None,
                  params_ineq: None = None):
    """Initialize KKT tuple."""
    # TODO(lbethune): implement other init schemes such as the ones of [4].
    # [4] Hafshejani, S.F. and Moaberfard, Z., 2021.
    # Initialization for Nonnegative Matrix Factorization: a Comprehensive Review.
    # arXiv preprint arXiv:2109.03874.
    Y = params_obj
    if self.init.startswith('sklearn_'):
      sklearn_init = self.init.split('_')[1]
      H1, H2_T = sklearn_initialize_nmf(Y, n_components=self.rank, init=sklearn_init, random_state=42)
      H2 = jnp.array(H2_T.T)
    else:
      raise ValueError(f'Unknown init {self.init}')
    U1, U2 = jnp.zeros_like(H1), jnp.zeros_like(H2)
    params = base.KKTSolution(primal=(H1, H2),
                              dual_eq=None,
                              dual_ineq=(U1, U2))
    return params

  def init_state(self,
                 init_params: base.KKTSolution,
                 params_obj: jnp.array,
                 params_eq: None = None,
                 params_ineq: None = None):
    Y            = params_obj
    kkt_sol_1, kkt_sol_2 = self._split_kkt_sol(init_params)
    nnls_state_1 = self.nnls_solver.init_state(kkt_sol_1, (Y.T, kkt_sol_2.primal))
    nnls_state_2 = self.nnls_solver.init_state(kkt_sol_2, (Y,   kkt_sol_1.primal))
    return NMFState(
        iter_num=jnp.asarray(0, dtype=jnp.int32),
        error=jnp.asarray(jnp.inf),
        nnls_state_1=nnls_state_1,
        nnls_state_2=nnls_state_2,
    )

  def update(self, params, state, params_obj, params_eq, params_ineq):
    """Update state of NMF.
    
    n: number of rows
    m: number of columns
    k: rank of low rank factorization

    Args:
      params: KKTSolution tuple, with params.primal = (H1, H2),
              H1 of shape (n, k), and H2 of shape (m, k)
      state: NMFState object.
      params_obj: Y of shape (n, m).
      params_eq: None, present for signature purposes.
      params_ineq: None, present for signature purposes.

    Returns:
      pair params, 
    """
    del params_eq, params_ineq  # unused
    H1, H2 = params.primal
    Y = params_obj

    kkt_sol_1, kkt_sol_2 = self._split_kkt_sol(params)

    # Solve \|Y.T - H2 H1.T\| = \|Y - H1 H2.T\| for H1
    kkt_sol_1, nnls_state_1 = self.nnls_solver.run(kkt_sol_1, (Y.T, kkt_sol_2.primal))

    # Solve \|Y - H1 H2.T\| for H2
    kkt_sol_2, nnls_state_2 = self.nnls_solver.run(kkt_sol_2, (Y  , kkt_sol_1.primal))

    H1, H2 = kkt_sol_1.primal, kkt_sol_2.primal
    error = frobenius_norm_squared(Y - H1 @ H2.T) / frobenius_norm_squared(Y)

    next_params = self._merge_kkt_sol(kkt_sol_1, kkt_sol_2)

    next_state = NMFState(
        iter_num=state.iter_num+1,
        error=error,
        nnls_state_1=nnls_state_1,
        nnls_state_2=nnls_state_2,
    )
    
    return base.OptStep(next_params, next_state)

  def run(self,
          params: Optional[base.KKTSolution],
          params_obj: jnp.array,
          params_eq: None = None,
          params_ineq: None = None):
    if params is None:
      params = self.init_params(params_obj, params_eq, params_ineq)
    return super().run(params, params_obj, params_eq, params_ineq)

  def __post_init__(self):
    self.optimality_fun = _make_nmf_kkt_optimality_fun()