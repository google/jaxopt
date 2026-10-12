import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_scipy_wrappers_module():
    module_path = Path(__file__).resolve().parents[1] / "jaxopt" / "_src" / "scipy_wrappers.py"

    jax_module = types.ModuleType("jax")
    jax_module.config = types.SimpleNamespace(jax_enable_x64=False)
    jax_module.jit = lambda fn: fn
    jax_module.hessian = lambda fn: (
        lambda x, *args, **kwargs: np.eye(np.asarray(x).size, dtype=np.asarray(x).dtype)
    )

    jax_numpy_module = types.ModuleType("jax.numpy")
    jax_numpy_module.asarray = np.asarray
    jax_numpy_module.array = np.array
    jax_numpy_module.ndarray = np.ndarray
    jax_numpy_module.float32 = np.float32
    jax_numpy_module.float64 = np.float64
    jax_numpy_module.einsum = np.einsum

    tree_util_module = types.ModuleType("jax.tree_util")
    tree_util_module.register_pytree_node_class = lambda cls: cls
    tree_util_module.tree_leaves = lambda x: [x]
    tree_util_module.tree_flatten = lambda x: ([x], None)
    tree_util_module.tree_unflatten = lambda treedef, leaves: leaves[0]
    tree_util_module.tree_map = lambda func, x: func(x)

    base_module = types.ModuleType("jaxopt._src.base")

    class Solver:
        pass

    class OptStep(tuple):
        def __new__(cls, params, info):
            return super().__new__(cls, (params, info))

    def _make_funs_without_aux(fun, value_and_grad, has_aux):
        grad_fun = lambda x, *args, **kwargs: x
        value_and_grad_fun = lambda x, *args, **kwargs: (0.0, x)
        return fun, grad_fun, value_and_grad_fun

    base_module.Solver = Solver
    base_module.OptStep = OptStep
    base_module.NUM_EVAL_DTYPE = np.int32
    base_module._make_funs_without_aux = _make_funs_without_aux

    idf_module = types.ModuleType("jaxopt._src.implicit_diff")
    idf_module.custom_root = lambda *args, **kwargs: (lambda fn: fn)

    projection_module = types.ModuleType("jaxopt._src.projection")
    tree_util_helpers_module = types.ModuleType("jaxopt._src.tree_util")
    tree_util_helpers_module.tree_sub = lambda x, y: x - y

    jaxopt_module = types.ModuleType("jaxopt")
    jaxopt_src_module = types.ModuleType("jaxopt._src")

    modules = {
        "jax": jax_module,
        "jax.numpy": jax_numpy_module,
        "jax.tree_util": tree_util_module,
        "jaxopt": jaxopt_module,
        "jaxopt._src": jaxopt_src_module,
        "jaxopt._src.base": base_module,
        "jaxopt._src.implicit_diff": idf_module,
        "jaxopt._src.projection": projection_module,
        "jaxopt._src.tree_util": tree_util_helpers_module,
    }
    previous_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)

    try:
        spec = importlib.util.spec_from_file_location("jaxopt_scipy_wrappers_under_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def test_trust_ncg_passes_hessian_to_scipy():
    scipy_wrappers = _load_scipy_wrappers_module()
    captured_hess = None

    def fake_minimize(fun, x0, jac, tol, bounds, method, callback, options, hess=None):
        nonlocal captured_hess
        captured_hess = hess
        return types.SimpleNamespace(
            x=np.asarray([3.0, 4.0]),
            fun=0.0,
            success=True,
            status=0,
            nit=1,
            nfev=1,
            njev=1,
            nhev=1,
        )

    scipy_wrappers.pytree_topology_from_example = lambda init_params: None
    scipy_wrappers.make_onp_to_jnp = lambda topology: (lambda x: x)
    scipy_wrappers.jnp_to_onp = lambda x, dtype=None: np.asarray(x, dtype=dtype)
    scipy_wrappers.osp.optimize.minimize = fake_minimize

    solver = scipy_wrappers.ScipyMinimize(
        fun=lambda x: np.sum(x**2),
        method="trust-ncg",
        jit=False,
    )

    solver._run(np.asarray([0.0, 0.0]), None)

    assert callable(captured_hess)
