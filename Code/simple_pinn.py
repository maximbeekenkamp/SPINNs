import time
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.example_libraries import optimizers
from jax.nn import tanh
from tqdm import trange
import matplotlib.pyplot as plt


"""
A Simple PINN is a partial derivative solver that is bound by physical laws, thus improving 
the accuracy of the solutions. 

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
def net_u(params, X, Y):
    """
    Defines neural network for u(x, y).

    Args:
        params (Tracer of Tuple[list[DeviceArray[float]]]): Lists containing weights and biases.
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.

    Returns:
        (Tracer of) DeviceArray: u(x, y).
    """
    xy_array = jnp.array([X, Y])
    return predict(params, xy_array)


@jit
def net_u_grad(params, X, Y):
    """
    Defines neural network for u'(x, y).

    Args:
        params (Tracer of Tuple[list[DeviceArray[float]]]): Lists containing weights and biases.
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.

    Returns:
        (Tracer of) DeviceArray: u'(x, y).
    """
    x_array = jnp.array([X, Y])
    y_pred = predict(params, x_array)
    return y_pred[0]


def net_f(params):
    """
    Defines neural network for first spatial derivative of u(x): u'(x).

    Args:
        params (Tracer of Tuple[list[DeviceArray[float]]]): Lists containing weights and biases.

    Returns:
        Tuple: Tuple containing functions for the first spatial derivative of u(x, y) with respect to x
        and secondly to y.
    """

    def ux(X, Y):
        """
        Calculates the first spatial derivative of u(x, y) with respect to x.

        Args:
            X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
            d-dimensional space.
            Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
            d-dimensional space.

        Returns:
            DeviceArray: First spatial derivative of u(x, y) with respect to x of shape (n, 1).
        """
        return grad(net_u_grad, argnums=1)(params, X, Y)

    def uy(X, Y):
        """
        Calculates the first spatial derivative of u(x, y) with respect to y.

        Args:
            X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
            d-dimensional space.
            Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
            d-dimensional space.

        Returns:
            DeviceArray: First spatial derivative of u(x, y) with respect to y of shape (n, 1).
        """
        return grad(net_u_grad, argnums=2)(params, X, Y)

    return jit(ux), jit(uy)


def net_ff(params):
    """
    Defines neural network for second spatial derivative of u(x, y): u''(x, y).

    Args:
        params (Tracer of Tuple[list[DeviceArray[float]]]): Lists containing weights and biases.

    Returns:
        Tuple: Tuple containing functions for the second spatial derivative of u(x, y) with respect to x
        and secondly to y..
    """

    def uxx(X, Y):
        """
        Calculates the second spatial derivative of u(x, y) with respect to x.

        Args:
            X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
            d-dimensional space.
            Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
            d-dimensional space.

        Returns:
            DeviceArray: Second spatial derivative of u(x, y) with respect to x of shape (n, 1).
        """
        u_x, _ = net_f(params)
        return grad(u_x, argnums=0)(X, Y)

    def uyy(X, Y):
        """
        Calculates the second spatial derivative of u(x, y) with respect to y.

        Args:
            X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
            d-dimensional space.
            Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
            d-dimensional space.

        Returns:
            DeviceArray: Second spatial derivative of u(x, y) with respect to y of shape (n, 1).
        """
        _, u_y = net_f(params)
        return grad(u_y, argnums=1)(X, Y)

    return jit(uxx), jit(uyy)


@jit
def net_bigu(params, X, Y):
    """
    Computes the bilinear form B(u,v) for a given neural network solution u(x) and
    input vectors X and Y.

    Args:
        params (Tracer of Tuple[list[DeviceArray[float]]]): Lists containing weights and biases.
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.

    Returns:
        (Tracer of) DeviceArray: The bilinear form B(u,v) evaluated at each pair of input points 
        (x_i, y_j), where u(x_i) is the neural network solution at x_i and v(y_j) is the neural 
        network solution at y_j.
    """
    u_xxf, u_yyf = net_ff(params)
    u_xx = vmap(vmap(u_xxf, in_axes=(None, 0)), in_axes=(0, None))(X, Y)
    u_yy = vmap(vmap(u_yyf, in_axes=(None, 0)), in_axes=(0, None))(X, Y)
    laplace = u_xx + u_yy
    return laplace


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
        4 * 10**6 * X**2
        + -4 * 10**6 * X
        + 4 * 10**6 * Y**2
        - 4 * 10**6 * Y
        + 1.996 * 10**6
    ) * jnp.exp(-1000 * ((X**2 - X + Y**2 - Y + 0.5)))


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
def loss(params, X, Y, bound, bfilter, ref):
    """
    Calculates the residual, boundary loss, and the L_2 error.

    Args:
        params (Tracer of Tuple[list[DeviceArray[float]]]): Lists containing weights and biases.
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
    u = vmap(vmap(net_u, in_axes=(None, None, 0)), in_axes=(None, 0, None))(
        params, X, Y
    )
    laplace = net_bigu(params, X, Y)
    fxy = vmap(vmap(funxy, in_axes=(None, 0)), in_axes=(0, None))(X, Y)
    res = laplace - fxy
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
    return jnp.sum((values * bfilter - bound).flatten() ** 2) / (
        2 * len(values[0]) + 2 * len(values) - 4
    )


@jit
def step(istep, opt_state, X, Y, bound, bfilter, ref):
    """
    Performs a single optimization step, computing gradients for network weights and applying the Adam
    optimizer to the network.

    Args:
        istep (int): Current iteration step number.
        opt_state (Tracer of OptimizerState): Optimised network parameters.
        X (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        Y (Tracer of DeviceArray): An array of shape (n, d) representing n collocation points in 
        d-dimensional space.
        bound (Tracer of DeviceArray): Boundary conditions.
        bfilter (Tracer of DeviceArray): The boundary condition filter.
        ref (Tracer of DeviceArray): The reference solution.

    Returns:
        (Tracer of) OptimizerState: An OptimizerState instance containing updated
        parameters for x, y coordinates.
    """
    params = get_params(opt_state)
    g = grad(loss, argnums=0, has_aux=True)(params, X, Y, bound, bfilter, ref)
    return opt_update(istep, g[0], opt_state)


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
    bound = np.array([[0 for a in range(len(X))] for b in range(len(Y))])
    bfilter = np.array([[0 for a in range(len(X) - 2)] for b in range(len(Y) - 2)])
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
x = jnp.arange(-1, 1.02, 0.02)
y = jnp.arange(-1, 1.02, 0.02)


"""
Model Hyperparameter initalisation.

Defined hyperparameters: 
    nu (float): Multiplicative constant.
    layer_sizes (list[int]): Network architecture.
    nIter (int): Number of epochs / iterations.
"""

layer_sizes = [2, 20, 20, 20, 1]
nIter = 10000 + 1

"""
Initialising weights, biases.

Weights and Biases:
    params (list[DeviceArray[float]]): Initialised weights and biases.
"""

params = init_network_params(layer_sizes, random.PRNGKey(52731))

"""
Initialising optimiser for weights/biases.

Optimiser:
    opt_state (list[DeviceArray[float]]): Initialised optimised weights and biases state.
"""

opt_init, opt_update, get_params = optimizers.adam(5e-4)
opt_state = opt_init(params)

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
    opt_state = step(it, opt_state, x, y, bound, bfilter, ref)
    if it % 1 == 0:
        params = get_params(opt_state)
        loss_full, losses = loss(params, x, y, bound, bfilter, ref)
        l_b = losses[1]
        l_f = losses[0]
        l2 = losses[2]

        pbar.set_postfix({"Loss": loss_full, "L2": l2})
        lb_list.append(l_b)
        lf_list.append(l_f)
        l2_list.append(l2)

end = time.time()
print(f"Runtime: {((end-start)/nIter*1000):.2f} ms/iter.")

u_pred = vmap(vmap(net_u, in_axes=(None, None, 0)), in_axes=(None, 0, None))(
    params, x, y
)

# print("lb",lb_list)
# print('lf', lf_list)

#######################################################
###                     PLOTTING                    ###
#######################################################
fig, axs = plt.subplots(1, 3, figsize=(24, 8))

shw = axs[0].imshow(u_pred, cmap="ocean")
axs[0].set_title("PINN Proposed Solution")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
fig.colorbar(shw, ax=axs[0])

axs[1].plot(lb_list, label="Boundary loss")
axs[1].plot(lf_list, label="Residue loss")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Loss")
axs[1].legend()
axs[1].set_title("Loss vs. Epochs")

axs[2].plot(l2_list, label="L2 Error")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Error")
axs[2].legend()
axs[2].set_title("L2 Error vs. Epochs")

plt.show()
