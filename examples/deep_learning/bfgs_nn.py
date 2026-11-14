import jax
import jax.numpy as jnp
from typing import Callable, Iterable

def wolfe(fun: Callable,
          res0, 
          x_0: Iterable, 
          dk, 
          c1: float=0.3, 
          c2: float=0.5, 
          alphas: float=0., 
          alphae: float=2.,
          eps: float=1e-3) -> float:
    assert c1 > 0
    assert c1 < 1
    assert c2 > 0
    assert c2 < 1
    assert c1 < c2
    assert alphas >= 0
    assert alphas < alphae
    assert eps > 0
    alpha = 1
    f0 = fun(x_0)
    while alphas < alphae:
        x = x_0 + (alpha*dk)[0]
        f1 = fun(x)
        if f1 <= f0 + c1*alpha*res0.dot(dk.T):
            res1 = jax.grad(fun)(x)
            if res1.dot(dk.T) >= c2*res0.dot(dk.T):
                break
            else:
                alphas = alpha
                alpha = 0.5 * (alphas + alphae)
        else:
            alphae = alpha
            alpha = 0.5 * (alphas + alphae)
        if jax.numpy.abs(alphas - alphae) < eps:
            break
    return alpha

def h2h(hessian):
    l = hessian.shape[0] # hessian.shape = (l, l)
    while 1:
        rank = jnp.linalg.matrix_rank(hessian)
        if rank == l:
            break
        else:
            hessian += jnp.identity(l)
    return hessian

def bfgs(fun: Callable, 
         x_0: Iterable,
         verbose: bool=False,
         epsilon: float=1e-6, 
         k: int=0):
    hessian = jax.hessian(fun)(x_0)
    hessian = h2h(hessian)
    while 1:
        gradient = jax.grad(fun)(x_0)
        dk = -jax.numpy.linalg.inv(hessian).dot(gradient.T).reshape(1, -1)
        if verbose:
            print('{}\t{}\t{}'.format(x_0, fun(x_0), k))
        if jax.numpy.linalg.norm(dk) >= epsilon:
            alpha = wolfe(fun, gradient, x_0, dk)
            delta = alpha * dk # sk, same as learning_date * grad(corrected by hessian and jax.grad)
            x_0 += delta[0]
            yk = jax.grad(fun)(x_0) - gradient
            s = delta.reshape(-1)                    
            sTy = float(s.dot(yk))                   
            if jnp.all(jnp.isfinite(yk)) and jnp.isfinite(sTy) and sTy > 1e-12:
                Hs = hessian.dot(delta.T)
                sTHs = float(s.dot(Hs.reshape(-1)))
                if jnp.isfinite(sTHs) and sTHs > 1e-12:
                    hessian += jnp.outer(yk, yk) / sTy - Hs.dot(Hs.T) / sTHs
            else:
                hessian = h2h(jnp.eye(hessian.shape[0], dtype=hessian.dtype))
            k += 1
        else:
            break
    return x_0

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

    def make_f_fun(Xb, yb):
        Xb = jnp.asarray(Xb).astype(dtype)
        yb_e = encode_y(yb)
        def f_batch(theta):
            return loss(theta, Xb, yb_e)
        return f_batch

    return f_fun, theta0, make_f_fun

def train_epochs(make_f_fun, f_fun, theta, X, y,
                 epochs=3, batch_size=128, epsilon=1e-1,
                 seed=0, verbose=True,
                 patience=10, min_delta=1e-8, monitor='batch'):
    if monitor not in ('batch', 'full'):
        raise ValueError("monitor must be 'batch' or 'full'.")

    X = jnp.asarray(X)
    y = jnp.asarray(y)
    n = X.shape[0]
    history = []
    early_stopped = False
    n_batches = (n + batch_size - 1) // batch_size

    for epoch in range(int(epochs)):
        perm = jax.random.permutation(jax.random.PRNGKey(seed + epoch), n)
        n_updated = 0
        no_decrease = 0
        prev_full = float('inf')

        for b, start in enumerate(range(0, n, batch_size)):
            idx = perm[start:start + batch_size]
            f_batch = make_f_fun(X[idx], y[idx])

            loss_before = float(f_batch(theta))
            theta = bfgs(f_batch, theta, verbose=False, epsilon=epsilon)
            loss_after = float(f_batch(theta))
            loss_full = float(f_fun(theta))

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
                      '{:.6f} -> {:.6f} | full loss {:.6f}'
                      .format(epoch, b, int(idx.shape[0]),
                              loss_before, loss_after, loss_full))

            if patience and no_decrease >= patience:
                early_stopped = True
                if verbose:
                    print('early stop | epoch {} | batch {:>2d} | {} consecutive batches '
                          'without loss decrease | full loss {:.6f}\n'
                          .format(epoch, b, patience, loss_full))
                break

        if verbose and not early_stopped:
            print('epoch {} done | full loss {:.6f} | updated {}/{} batches\n'
                  .format(epoch, float(f_fun(theta)), n_updated, n_batches))
        if early_stopped:
            break

    return theta, history


if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    import pandas, numpy
    df = pandas.read_csv('https://raw.githubusercontent.com/linjing-lab/easy-pytorch/main/released_box/data/bitcoin_heist_data.csv', engine="pyarrow")
    df = df.to_numpy()
    labels = df[:,-1]
    X = df[:,1:-1].astype(numpy.float64)
    unique = numpy.unique(labels)
    indices = dict(zip(unique, range(len(unique))))
    y = numpy.array([indices[value] for value in labels], dtype=numpy.int8)

    jax.config.update("jax_enable_x64", True)
    EPOCHS = 1
    BATCH_SIZE = 512
    EPSILON = 0.1
    SEED = 42
    X = StandardScaler().fit_transform(X)
    hidden = [40,]
    print(X.shape, y.shape)

    f_fun, theta0, make_f_fun = gen_nn(X, y, hidden_dims=hidden, task="classification", seed=SEED)

    f_jit = jax.jit(f_fun)
    print('init:', f_jit(theta0), jax.grad(f_jit)(theta0).shape)

    theta, hist = train_epochs(
        make_f_fun, f_fun, theta0, X, y,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        epsilon=EPSILON, seed=SEED, verbose=True,
        patience=10, monitor='batch')

    print('after {} epochs, full loss = {}'.format(EPOCHS, float(f_fun(theta))))
    per_epoch = {}
    for h in hist:
        per_epoch[h[0]] = h[5]
    print('epoch-end full loss:', [round(per_epoch[e], 6) for e in range(EPOCHS)])
