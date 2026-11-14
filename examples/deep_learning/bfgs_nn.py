# -*- coding: utf-8 -*-
import inspect
import os
from typing import Callable

import jax
import jax.numpy as jnp

import jaxopt
jax.config.update("jax_enable_x64", True)

def make_solver(fun: Callable,
                method: str = 'lbfgs',
                maxiter: int = 30,
                tol: float = 1e-1,
                c1: float = 0.3,
                c2: float = 0.5,
                maxls: int = 20,
                verbose: bool = False):
    solvers = {'lbfgs': jaxopt.LBFGS, 'bfgs': jaxopt.BFGS}
    if method not in solvers:
        raise ValueError(f"method must be one of {list(solvers)}, got {method!r}")
    cls = solvers[method]

    allowed = set(inspect.signature(cls.__init__).parameters)
    opts = dict(fun=fun,
                maxiter=maxiter,
                tol=tol,
                linesearch='zoom',
                maxls=maxls,
                jit=True,
                verbose=verbose)

    if 'c1' in allowed:
        opts.update(c1=c1, c2=c2, condition='strong-wolfe')
    elif c1 is not None and c2 is not None and hasattr(jaxopt, 'ZoomLineSearch'):
        opts['linesearch'] = jaxopt.ZoomLineSearch(
            fun=fun, c1=c1, c2=c2, maxiter=maxls, jit=True)
    opts = {k: v for k, v in opts.items() if k in allowed}
    return cls(**opts)

def xavier_normal(layer_dims, key, dtype=jnp.float64):
    layer_dims = [int(d) for d in layer_dims]
    chunks = []
    for i, (fan_in, fan_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        std = jnp.sqrt(2.0 / (fan_in + fan_out))
        sub = jax.random.fold_in(key, i)
        chunks.append(jax.random.normal(sub, (fan_in * fan_out,), dtype=dtype) * std)
        chunks.append(jnp.zeros((fan_out,), dtype=dtype))
    return jnp.concatenate(chunks)


def gen_nn(X_data, y_data, hidden_dims=(), task='classification', key=None, seed=0):
    if key is None:
        key = jax.random.PRNGKey(seed)
    X_data = jnp.asarray(X_data)
    y_data = jnp.asarray(y_data)
    dtype = jnp.result_type(X_data.dtype, jnp.float32)
    X_data = X_data.astype(dtype)

    _, n_features = X_data.shape
    n_features = int(n_features)

    classes = None
    if task == 'classification':
        if y_data.ndim == 1:
            classes = jnp.unique(y_data)
            n_outputs = int(classes.size)
        else:
            n_outputs = int(y_data.shape[1])
    elif task == 'regression':
        n_outputs = 1 if y_data.ndim == 1 else int(y_data.shape[1])
    else:
        raise ValueError(f"Support classification or regression. not support {task}.")

    def encode_y(y_batch):
        y_batch = jnp.asarray(y_batch)
        if task == 'regression':
            out = y_batch.reshape(-1, 1) if y_batch.ndim == 1 else y_batch
            return out.astype(dtype)
        if y_batch.ndim == 1:
            return jnp.eye(n_outputs, dtype=dtype)[jnp.searchsorted(classes, y_batch)]
        return y_batch.astype(dtype)

    layer_dims = jnp.concatenate([
        jnp.asarray([n_features]),
        jnp.asarray(hidden_dims, dtype=jnp.int32).reshape(-1),
        jnp.asarray([n_outputs]),
    ])
    layer_dims = [int(d) for d in layer_dims]
    sizes = list(zip(layer_dims[:-1], layer_dims[1:]))

    theta0 = xavier_normal(layer_dims, key, dtype=dtype)

    cuts, acc = [], 0
    for prev_dim, curr_dim in sizes:
        acc += prev_dim * curr_dim
        cuts.append(acc)
        acc += curr_dim
        cuts.append(acc)
    boundaries = cuts[:-1]

    def unpack(theta):
        parts = jnp.split(theta.reshape(-1), boundaries)
        Ws = [parts[2 * i].reshape(curr_dim, prev_dim)
              for i, (prev_dim, curr_dim) in enumerate(sizes)]
        bs = [parts[2 * i + 1].reshape(curr_dim)
              for i, (prev_dim, curr_dim) in enumerate(sizes)]
        return Ws, bs

    def forward(theta, X):
        Ws, bs = unpack(theta)
        H = X
        for W, b in zip(Ws, bs):
            H = H @ W.T + b
        return H

    def loss(theta, Xb, yb_e):
        logits = forward(theta, Xb)
        if task == 'classification':
            lse = jax.nn.logsumexp(logits, axis=1)
            return jnp.mean(lse - jnp.sum(yb_e * logits, axis=1))
        d = logits - yb_e
        return jnp.sum(d * d) / (Xb.shape[0] * n_outputs)

    y_full = encode_y(y_data)

    def f_fun(theta):
        return loss(theta, X_data, y_full)

    def loss_fn(theta, Xb, yb):
        Xb = jnp.asarray(Xb).astype(dtype)
        return loss(theta, Xb, encode_y(yb))

    return f_fun, theta0, loss_fn

def train_epochs(loss_fn, f_fun, theta, X, y,
                 epochs=3, batch_size=128, epsilon=1e-1,
                 seed=0, verbose=True,
                 patience=10, min_delta=1e-8, monitor='batch',
                 method='lbfgs', maxiter=30,
                 c1=0.3, c2=0.5, maxls=20,
                 solver_verbose=False):
    if monitor not in ('batch', 'full'):
        raise ValueError("monitor must be 'batch' or 'full'.")

    X = jnp.asarray(X)
    y = jnp.asarray(y)
    n = X.shape[0]
    history = []
    early_stopped = False
    n_batches = (n + batch_size - 1) // batch_size

    solver = make_solver(loss_fn, method=method, maxiter=maxiter, tol=epsilon,
                         c1=c1, c2=c2, maxls=maxls, verbose=solver_verbose)
    eval_loss = jax.jit(loss_fn)
    full_loss = jax.jit(f_fun)

    for epoch in range(int(epochs)):
        perm = jax.random.permutation(jax.random.PRNGKey(seed + epoch), n)
        n_updated = 0
        no_decrease = 0
        prev_full = float('inf')

        for b, start in enumerate(range(0, n, batch_size)):
            idx = perm[start:start + batch_size]
            Xb, yb = X[idx], y[idx]

            loss_before = float(eval_loss(theta, Xb, yb))
            step = solver.run(theta, Xb, yb)
            theta = step.params
            loss_after = float(eval_loss(theta, Xb, yb))
            loss_full = float(full_loss(theta))

            if monitor == 'full':
                decreased = loss_full < prev_full - min_delta
                prev_full = loss_full
            else:
                decreased = loss_after < loss_before - min_delta

            no_decrease = 0 if decreased else no_decrease + 1
            if abs(loss_after - loss_before) > 1e-12:
                n_updated += 1

            history.append((epoch, b, int(idx.shape[0]),
                            loss_before, loss_after, loss_full))
            if verbose:
                print('epoch {} | batch {:>2d} | n={:<3d} | batch loss '
                      '{:.6f} -> {:.6f} | full loss {:.6f} | {} iters | grad norm {:.3e}'
                      .format(epoch, b, int(idx.shape[0]),
                              loss_before, loss_after, loss_full,
                              int(step.state.iter_num), float(step.state.error)))

            if patience and no_decrease >= patience:
                early_stopped = True
                if verbose:
                    print('early stop | epoch {} | batch {:>2d} | {} consecutive batches '
                          'without loss decrease | full loss {:.6f}\n'
                          .format(epoch, b, patience, loss_full))
                break

        if verbose and not early_stopped:
            print('epoch {} done | full loss {:.6f} | updated {}/{} batches\n'
                  .format(epoch, float(full_loss(theta)), n_updated, n_batches))
        if early_stopped:
            break

    return theta, history


if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    import pandas
    import numpy
    df = pandas.read_csv("https://raw.githubusercontent.com/linjing-lab/easy-pytorch/main/released_box/data/bitcoin_heist_data.csv", engine="pyarrow").to_numpy()
    labels = df[:, -1]
    X = df[:, 1:-1].astype(numpy.float64)
    unique = numpy.unique(labels)
    indices = dict(zip(unique, range(len(unique))))
    y = numpy.array([indices[value] for value in labels], dtype=numpy.int8)

    EPOCHS = 1
    BATCH_SIZE = 512
    EPSILON = 0.1
    SEED = 42
    X = StandardScaler().fit_transform(X)
    hidden = [40, ]
    print(X.shape, y.shape)

    f_fun, theta0, loss_fn = gen_nn(
        X, y, hidden_dims=hidden, task="classification", seed=SEED)

    f_jit = jax.jit(f_fun)
    print('init:', f_jit(theta0), jax.grad(f_jit)(theta0).shape)

    theta, hist = train_epochs(
        loss_fn, f_fun, theta0, X, y,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        epsilon=EPSILON, seed=SEED, verbose=True,
        patience=10, monitor='batch',
        method='lbfgs',
        maxiter=30, c1=0.3, c2=0.5)

    print('after {} epochs, full loss = {}'.format(EPOCHS, float(f_fun(theta))))
    per_epoch = {}
    for h in hist:
        per_epoch[h[0]] = h[5]
    print('epoch-end full loss:', [round(per_epoch[e], 6) for e in range(EPOCHS)])
