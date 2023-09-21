# Copyright 2023 Google LLC
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

import dataclasses
from typing import Any, Callable, NamedTuple, Optional, Union

import jax.numpy as jnp
import jaxopt
from jaxopt._src import base
from jaxopt._src.linesearch_util import _init_stepsize, _setup_linesearch
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_negative
from jaxopt._src.tree_util import tree_single_dtype
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_zeros_like


class SolverWithLineSearchState(NamedTuple):
  iter_num: int
  error: float
  value: float
  stepsize: float
  velocity: Optional[Any]
  aux: Optional[Any]


@dataclasses.dataclass(eq=False)
class SolverWithLineSearch(base.StochasticSolver):
  """Solver computing next iterate by line-search along update direction.
  
  Args:
    fun: objective function
    value_and_grad_fun: If `value_and_grad` is a boolean, determines whether
      the function returns both balue and gradient or just the value. 
      If `value_and_grad` is a Callable, `fun` should return value
      only, and `value_and_grad` returns value and gradient.
    update_dir_fun: function update direction along which to search. 
      If None the update direction is the negative gradient.
    has_aux: whether function has an auxiliary output

    init_stepsize: initial stepsize
    linesearch: linesearch to use. Must be an IterativeLinesearch class. 
      To use some specific options for a linesearch like zoom, use e.g.
      linesearch= functools.partial(ZoomLineSearch, tol=1e-3) 
      (which gives a zoom linesearch with a loose tolerance)
    increase_factor: factor by which to increase the stepsize during line search
      (default: 1.5).
    min_stepsize: lower bound on stepsize.
    max_stepsize: upper bound on stepsize.
    linesearch_init: strategy for line-search initialization. By default, it
      will use "increase", which will increase the step-size by a factor of
      `increase_factor` at each iteration if the step-size is larger than
      `min_stepsize`, and set it to `max_stepsize` otherwise. Other choices are
      "max", that initializes the step-size to `max_stepsize` at every
      iteration, and "current", that uses the step-size from the previous
      iteration.
    momentum: momentum parameter
    max_iter: maximal number of iterations
    tol: tolerance of the stopping criterion
    verbose: whether to print error on every iteration or not.
    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").
  """
  fun: Callable
  value_and_grad: Union[bool, Optional[Callable]] = None
  update_dir_fun: Optional[Callable] = None
  has_aux: bool = False

  init_stepsize: float = 1.0
  linesearch: base.IterativeLineSearch = jaxopt.BacktrackingLineSearch
  increase_factor: float = 1.5
  min_stepsize: float = 1e-6
  max_stepsize: float = 1.0
  linesearch_init: str = 'increase'

  momentum: float = 0.0

  max_iter: int = 500
  tol: float = 0.

  verbose: bool = False

  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self, init_params: Any) -> SolverWithLineSearchState:
    if self.momentum == 0:
      velocity = None
    else:
      velocity = tree_zeros_like(init_params)

    param_dtype = tree_single_dtype(init_params)

    return SolverWithLineSearchState(
        iter_num=jnp.asarray(0),
        error=jnp.asarray(jnp.inf),
        value=jnp.asarray(jnp.inf),
        stepsize=jnp.asarray(self.init_stepsize, dtype=param_dtype),
        velocity=velocity,
        aux=None,
    )

  def update(
      self,
      params: Any,
      state: SolverWithLineSearchState,
      *args: Any,
      **kwargs: Any,
  ) -> base.OptStep:
    (value, aux), grad = self.value_and_grad_fun_with_aux(
        params, *args, **kwargs
    )

    if self.update_dir_fun is not None:
      update_direction = self.update_dir_fun(params, *args, **kwargs)
    else:
      update_direction = tree_negative(grad)

    init_stepsize = _init_stepsize(
        self.linesearch_init,
        self.max_stepsize,
        self.min_stepsize,
        self.increase_factor,
        state.stepsize,
    )
    stepsize, ls_state = self.run_ls(
        init_stepsize,
        params,
        value,
        grad,
        update_direction,
        fun_args=args,
        fun_kwargs=kwargs,
    )
    next_params = ls_state.params

    if self.momentum == 0:
      velocity = None
    else:
      next_params = tree_add_scalar_mul(
          next_params, self.momentum, state.velocity
      )
      velocity = tree_sub(next_params, params)

    new_state = SolverWithLineSearchState(
        iter_num=state.iter_num + 1,
        error=tree_l2_norm(grad),
        value=value,
        stepsize=stepsize,
        velocity=velocity,
        aux=aux,
    )
    return base.OptStep(params=next_params, state=new_state)

  def __post_init__(self):
    self.fun_with_aux, _, self.value_and_grad_fun_with_aux = (
        base._make_funs_with_aux(
            self.fun,
            self.value_and_grad,
            self.has_aux,
        )
    )

    linesearch_solver = self.linesearch(
      fun=self.fun,
      value_and_grad=self.value_and_grad_fun_with_aux,
      has_aux=True,
      jit=False,  # do not jit inside the overall jit
      # TODO(vroulet): not sure about what should be the default option for
      # unrolling
      unroll=self._get_unroll_option(),
      verbose=self.verbose
      )
    self.run_ls = linesearch_solver.run

