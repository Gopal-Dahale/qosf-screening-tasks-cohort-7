import pennylane as qml
import jax
import jax.numpy as jnp
from itertools import product

def square_kernel_matrix_jax(X, kernel, assume_normalized_kernel=False):
    N = qml.math.shape(X)[0]
    if assume_normalized_kernel and N == 1:
        return qml.math.eye(1, like=qml.math.get_interface(X))

    matrix = [None] * N**2
    

    # Compute all off-diagonal kernel values, using symmetry of the kernel matrix
    for i in range(N):
        for j in range(i + 1, N):
            matrix[N * i + j] = (kernel_value := kernel(X[i], X[j]))
            matrix[N * j + i] = kernel_value
    
    i, j = jnp.tril_indices(N)
    res = jax.vmap(kernel, in_axes=(0,0))(X[i], X[j])
    mtx = jnp.zeros((N, N))  # create an empty matrix
    mtx = mtx.at[jnp.tril_indices(N)].set(res)
    mtx = mtx + mtx.T - jnp.diag(jnp.diag(mtx))
    
    if assume_normalized_kernel:
        mtx = mtx.at[jnp.diag_indices_from(mtx)].set(1)
        
    return mtx

def kernel_matrix_jax(X1, X2, kernel):
    N = qml.math.shape(X1)[0]
    M = qml.math.shape(X2)[0]

    products = jnp.array(list(product(X1,X2)))
    mtx = qml.math.stack(jax.vmap(kernel, in_axes=(0,0))(products[:,0,:], products[:,1,:]))

    if qml.math.ndim(mtx[0]) == 0:
        return qml.math.reshape(mtx, (N, M))

    return qml.math.moveaxis(qml.math.reshape(mtx, (N, M, qml.math.size(mtx[0]))), -1, 0)

def target_alignment_jax(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    """Kernel-target alignment between kernel and labels."""

    K = square_kernel_matrix_jax(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    if rescale_class_labels:
        nplus = jnp.count_nonzero(jnp.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = jnp.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = jnp.array(Y)

    T = jnp.outer(_Y, _Y)
    inner_product = jnp.sum(K * T)
    norm = jnp.sqrt(jnp.sum(K * K) * jnp.sum(T * T))
    inner_product = inner_product / norm

    return inner_product