import time
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, jvp
from jax import random
from jax.example_libraries import optimizers
from jax.nn import tanh
from tqdm import trange
import matplotlib.pyplot as plt


"""
A Separable PINN is a partial derivative solver that improves on a simple PINN model by leveraging 
forward-mode autodifferentiation and operates on a per-axis basis. 

This 2D time independent implementation of a PINN solves a two-dimensional differential heat 
equation defined by: 

-Delta u() = 
(4*10^6 * x^2 - 2*10^6 * x + 4*10^6 * y^2 - 2*10^6 * y + 49600) * exp (-1000[(x_1 - r_c)^2 + (x_2 - r_c)^2]) 
on [-1,1]^2
and u() = g() on boundry
u(x_1, x_2) = exp (-1000[(x_1 - r_c)^2 + (x_2 - r_c)^2]) = g()

"""

#######################################################
###     FEEDFORWARD NEURAL NETWORK ARCHITECTURE     ###
#######################################################


def random_layer_params(m, n, key, scale):
    """
    An init_network_params helper function to randomly initialize weights and biases for a
    dense neural network layer.

    Args:
        m (int): Shape of our weights (MxN matrix).
        n (int): Shape of our weights (MxN matrix).
        key (DeviceArray): A random key used for initialisation.
        scale (DeviceArray[float]): A float value between 0 and 1, used to scale the initialisation.

    Returns:
        Tuple[DeviceArray]: Randomised initialisation for a single layer. This consists of the
        weight matrix with shape (m, n) and the bias vector with shape (n,).
    """
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (m, n)), jnp.zeros(n)


def init_network_params(sizes, key):
    """
    Initialises the parameters for a fully-connected neural network with the
    given network architecture of size "sizes".

    Args:
        sizes (List[int]): A list of integers representing the number of neurons in each layer of the 
        network, including the input and output layers.
        key (DeviceArray): A random key used for initialisation.

    Returns:
        List[Tuple[DeviceArray]]: Fully initialised network parameters. 
    """
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k, 2.0 / (jnp.sqrt(m + n)))
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


@jit
def predict(params, X):
    """
    Per example predictions.

    Args:
        params (Tracer of list[DeviceArray[float]]): List containing weights and biases.
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.

    Returns:
        (Tracer of) DeviceArray: Output predictions (u_pred).
    """
    activations = X
    for w, b in params[:-1]:
        activations = tanh(jnp.dot(activations, w) + b)
    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits


@jit
def net_u(params, X):
    """
    Defines neural network for u(x).

    Args:
        params (Tracer of list[DeviceArray[float]]): List containing weights and biases.
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.

    Returns:
        (Tracer of) DeviceArray: u(x).
    """
    x_array = jnp.array([X])
    return predict(params, x_array)


def hvp_fwdfwd(f, primals, tangents):
    """
    Compute the Hessian-vector product (HVP) of a given function f with respect to its
    inputs primals and tangents, using forward-mode differentiation.

    Args:
        f (Callable): net_bigu
        primals (Tuple): The input array of shape (n, d), contains only a single set of axes.
        tangents (Tuple): An array of shape (n, d) of 'ones'.

    Returns:
        Tuple: The i-th element of the tuple represents the HVP of f with respect to the i-th 
        element of primals and the corresponding element of tangents.
    """
    g = lambda primals: jvp(f, (primals,), tangents)[1]
    primals_out, tangents_out = jvp(g, primals, tangents)
    return tangents_out


@jit
def net_bigu(x_params, y_params, X, Y):
    """
    Computes the bilinear form B(u,v) for a given neural network solution u(x) and
    input vectors X and Y.

    Args:
        x_params (Tracer of list[DeviceArray[float]]): List containing weights and biases.
        y_params (Tracer of list[DeviceArray[float]]): List containing weights and biases.
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.

    Returns:
        (Tracer of) DeviceArray: The bilinear form B(u,v) evaluated at each pair of input points 
        (x_i, y_j), where u(x_i) is the neural network solution at x_i and v(y_j) is the neural 
        network solution at y_j.
    """
    u_x = vmap(net_u, (None, 0))(x_params, X)
    u_y = vmap(net_u, (None, 0))(y_params, Y)
    return jnp.sum(jnp.einsum("in,jn->nij", u_x, u_y), axis=0)


@jit
def net_laplace(x_params, y_params, X, Y):
    """
    Computes the Laplacian of a given neural network solution u(x,y) with respect to the
    input variables x and y.

    Args:
        x_params (Tracer of list[DeviceArray[float]]): List containing weights and biases.
        y_params (Tracer of list[DeviceArray[float]]): List containing weights and biases.
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.

    Returns:
        (Tracer of) DeviceArray: the Laplacian of the neural network solution u(x,y) evaluated at
        each pair of input points (x_i, y_j), where u(x_i,y_j) is the neural network solution at 
        (x_i,y_j).
    """
    v = jnp.ones(X.shape)
    u_xx = hvp_fwdfwd(lambda x: net_bigu(x_params, y_params, x, Y), (X,), (v,))
    u_yy = hvp_fwdfwd(lambda y: net_bigu(x_params, y_params, X, y), (Y,), (v,))
    return -(u_xx + u_yy)


@jit
def funxy(X, Y):
    """
    Returns the value of the function f(x,y ) at the given collocation points X and Y.

    Args:
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.

    Returns:
        (Tracer of) DeviceArray: An array of shape (n,) representing the value of the function at the 
        given collocation points.
    """
    return (
        -4 * 10**6 * X**2
        + 4 * 10**6 * X
        - 4 * 10**6 * Y**2
        + 4 * 10**6 * Y
        - 1.996 * 10**6
    ) * jnp.exp(-1000 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


@jit
def finalfunc(X, Y):
    """
    Returns the value of the final function at the given collocation points X and Y.

    Args:
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.

    Returns:
        (Tracer of) DeviceArray: An array of shape (n,) representing the value of the final function 
        at the given collocation points.
    """
    return jnp.exp(-1000 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


@jit
def loss(x_params, y_params, X, Y, bound, bfilter, ref):
    
    """
    Calculates the residual, boundary loss, and the L_2 error.

    Args:
        x_params (Tracer of list[DeviceArray[float]]): List containing weights and biases.
        y_params (Tracer of list[DeviceArray[float]]): List containing weights and biases.
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        bound (Tracer of DeviceArray): Boundary conditions.
        bfilter (Tracer of DeviceArray): The boundary condition filter.
        ref (Tracer of DeviceArray): The reference solution.


    Returns:
        (Tracer of) Tuple[DeviceArray]: Loss, Residual Loss, Boundary Loss, and L_2 error.
    """
    u = net_bigu(x_params, y_params, X, Y)
    u_laplace = net_laplace(x_params, y_params, X, Y)
    fxy = vmap(vmap(funxy, in_axes=(None, 0)), in_axes=(0, None))(X, Y)
    res = u_laplace - fxy
    lossb = loss_b(u, bound, bfilter)
    lossf = jnp.mean((res.flatten()) ** 2)
    loss = jnp.sum(lossf + lossb)
    l2 = jnp.linalg.norm(u - ref) / jnp.linalg.norm(ref)
    return (loss, (lossf, lossb, l2))


@jit
def loss_b(values, bound, bfilter):
    """
    Calculates the boundary loss.

    Args:
        values (Tracer of Tracer of DeviceArray): Predicted values.
        bound (DeviceArray): Boundary conditions.
        bfilter (DeviceArray):The boundary condition filter.

    Returns:
        (Tracer of) DeviceArray: Boundary loss.
    """
    return jnp.sum((values * bfilter - bound).flatten() ** 2) / (2 * len(values[0]) + 2 * len(values) - 4)


@jit
def step(istep, opt_state_x, opt_state_y, X, Y, bound, bfilter, ref):
    """
    Performs a single optimization step, computing gradients for network weights and applying the Adam
    optimizer to the network.

    Args:
        istep (int): Current iteration step number.
        opt_state_x (Tracer of OptimizerState): Optimised network parameters for x coordinates.
        opt_state_y (Tracer of OptimizerState): Optimised network parameters for y coordinates.
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        bound (Tracer of DeviceArray): Boundary conditions.
        bfilter (Tracer of DeviceArray): The boundary condition filter.
        ref (Tracer of DeviceArray): The reference solution.

    Returns:
        (Tracer of) Tuple[OptimizerState]: A tuple of two OptimizerState instances containing updated
        parameters for x and y coordinates respectively.
    """
    param_x = get_params_x(opt_state_x)
    param_y = get_params_y(opt_state_y)
    g_x = grad(loss, argnums=0, has_aux=True)(
        param_x, param_y, X, Y, bound, bfilter, ref
    )
    g_x = g_x[0]
    g_y = grad(loss, argnums=1, has_aux=True)(
        param_x, param_y, X, Y, bound, bfilter, ref
    )
    g_y = g_y[0]
    return opt_update_x(istep, g_x, opt_state_x), opt_update_y(istep, g_y, opt_state_y)


def setup_boundry(X, Y):
    """
    Sets up boundary conditions for the problem.

    Args:
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.

    Returns:
        Tuple[DeviceArray]: A tuple of the boundary conditions and the filter applied to them.
    """
    bound = np.array([[0 for _ in range(len(X))] for _ in range(len(Y))])
    bfilter = np.array([[0 for _ in range(len(X) - 2)] for _ in range(len(Y) - 2)])
    bfilter = np.pad(bfilter, ((1, 1), (1, 1)), constant_values=1)
    for y in range(len(Y)):
        bound[y][0] = finalfunc(X[0], Y[y])
        bound[y][len(X) - 1] = finalfunc(X[len(X) - 1], Y[y])
    for x in range(len(X)):
        bound[0][x] = finalfunc(X[x], Y[0])
        bound[len(Y) - 1][x] = finalfunc(X[x], Y[len(Y) - 1])
    return jnp.array(bound), jnp.array(bfilter)


#######################################################
###              MODEL HYPERPARAMETERS              ###
#######################################################


# Generation of 'input data', known as collocation points.
x = jnp.linspace(-1, 1, 100)
y = jnp.linspace(-1, 1, 100)


"""
Model Hyperparameter initalisation.

Defined hyperparameters: 
    nu (float): Multiplicative constant.
    layer_sizes (list[int]): Network architecture.
    nIter (int): Number of epochs / iterations.
"""
layer_sizes = [1, 20, 20, 20, 2]
nIter = 100000 + 1

"""
Initialising weights, biases.

Weights and Biases:
    params (list[DeviceArray[float]]): Initialised weights and biases.
"""

params_x = init_network_params(layer_sizes, random.PRNGKey(93256))
params_y = init_network_params(layer_sizes, random.PRNGKey(76148))

"""
Initialising optimiser for weights/biases.

Optimiser:
    opt_state (list[DeviceArray[float]]): Initialised optimised weights and biases state.
"""

opt_init_x, opt_update_x, get_params_x = optimizers.adam(5e-4)
opt_state_x = opt_init_x(params_x)

opt_init_y, opt_update_y, get_params_y = optimizers.adam(5e-4)
opt_state_y = opt_init_y(params_y)

opt_init_x2, opt_update_x2, get_params_x2 = optimizers.adam(1e-8)
opt_state_x2 = opt_init_x2(params_x)

opt_init_y2, opt_update_y2, get_params_y2 = optimizers.adam(1e-8)
opt_state_y2 = opt_init_y2(params_y)

# lists for boundary and residual loss values during training.
lb_list = []
lf_list = []
l2_list = []

#######################################################
###                  MODEL TRAINING                 ###
#######################################################
bound, bfilter = setup_boundry(x, y)
ref = vmap(vmap(finalfunc, in_axes=(None, 0)), in_axes=(0, None))(x, y)
pbar = trange(nIter)

start = time.time()
for it in pbar:
    if it < 500:
        opt_state_x, opt_state_y = step(
            it, opt_state_x, opt_state_y, x, y, bound, bfilter, ref
        )
        if it % 1 == 0:
            params_x = get_params_x(opt_state_x)
            params_y = get_params_y(opt_state_y)
            loss_full, losses = loss(params_x, params_y, x, y, bound, bfilter, ref)
            l_b = losses[1]
            l_f = losses[0]
            l2 = losses[2]

            pbar.set_postfix({"Loss": loss_full, "L2": l2})
            lb_list.append(l_b)
            lf_list.append(l_f)
            l2_list.append(l2)
    else:
        opt_state_x2, opt_state_y2 = step(
            it, opt_state_x2, opt_state_y2, x, y, bound, bfilter, ref
        )
        if it % 1 == 0:
            params_x2 = get_params_x2(opt_state_x2)
            params_y2 = get_params_y2(opt_state_y2)
            loss_full, losses = loss(params_x2, params_y2, x, y, bound, bfilter, ref)
            l_b = losses[1]
            l_f = losses[0]
            l2 = losses[2]

            pbar.set_postfix({"Loss": loss_full, "L2": l2})
            lb_list.append(l_b)
            lf_list.append(l_f)
            l2_list.append(l2)

end = time.time()
print(f"Runtime: {((end-start)/nIter*1000):.2f} ms/iter.")

u_pred = net_bigu(params_x, params_y, x, y)

#######################################################
###                     PLOTTING                    ###
#######################################################
fig, axs = plt.subplots(1, 3, figsize=(24, 8))

shw = axs[0].imshow(u_pred, cmap="ocean")
axs[0].set_title("SPINN Proposed Solution")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
fig.colorbar(shw, ax=axs[0])

axs[1].plot(lb_list, label="Boundary loss")
axs[1].plot(lf_list, label="Residue loss")
axs[1].set_xlabel("Epoch")
axs[1].set_yscale('log')
axs[1].set_ylabel("Loss")
axs[1].legend()
axs[1].set_title("Loss vs. Epochs")

axs[2].plot(l2_list, label="L2 Error")
axs[2].set_xlabel("Epoch")
axs[2].set_yscale('log')
axs[2].set_ylabel("Error")
axs[2].legend()
axs[2].set_title("L2 Error vs. Epochs")

# axs[1].plot(lf_list, label="Residue loss")
# axs[1].set_xlabel("Epoch")
# axs[1].set_yscale('log')
# axs[1].set_ylabel("Loss")
# axs[1].legend()
# axs[1].set_title("Loss vs. Epochs")

# axs[1].plot(lb_list, label="Boundary loss")
# axs[1].set_xlabel("Epoch")
# axs[1].set_yscale('log')
# axs[1].set_ylabel("Loss")
# axs[1].legend()
# axs[1].set_title("Loss vs. Epochs")


plt.show()
