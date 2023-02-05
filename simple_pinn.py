import jax.numpy as jnp
from jax import grad, jit, vmap, jvp
from jax import random
from jax.example_libraries import optimizers
from jax.nn import tanh
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


"""
A simple PINN is a partial derivative solver that is bound by physical laws, thus improving 
the accuracy of the solutions. 

This 2D time dependent implementation of a PINN solves a two-dimensional differential heat equation defined by: 

u(x_1, x_2) = exp (-1000[(x_1 - r_c)^2 + (x_2 - r_c)^2]) 


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
        key (DeviceArray): Key for random modules.
        scale (DeviceArray[float]): Float value between 0 and 1 to scale the initialisation.

    Returns:
        DeviceArray: Randomised initialisation for a single layer.
    """
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (m, n)), jnp.zeros(n)


def init_network_params(sizes, key):
    """
    Initializes all layers for a fully-connected neural network with
    size "sizes".

    Args:
        sizes (list[int]): Network architecture.
        key (DeviceArray): Key for random modules.

    Returns:
        list[DeviceArray]: Fully initialised network parameters.
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
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        (Tracer of) DeviceArray: Output predictions (u_pred).
    """
    activations = X
    for w, b in params[:-1]:
        activations = tanh(jnp.dot(activations, w) + b)
    final_w, final_b = params[-1]
    logits = jnp.sum(jnp.dot(activations, final_w) + final_b)
    print(logits.shape)
    return logits


@jit
def net_u(params, X):
    """
    Defines neural network for u(x).

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        (Tracer of) DeviceArray: u(x).
    """
    x_array = jnp.array([X])
    return predict(params, x_array)

"""

# Isolate the function from the weight matrix to the predictions
f = lambda W: predict(W, b, inputs)

key, subkey = random.split(key)
v = random.normal(subkey, W.shape)

# Push forward the vector `v` along `f` evaluated at `W`
y, u = jvp(f, (W,), (v,))

def vmap_jmp(f, W, M):
    _jvp = lambda s: jvp(f, (W,), (s,))[1]
    return vmap(_jvp)(M)

"""

def net_ux(params):
    """
    Defines neural network for first spatial derivative of u(x): u'(x).

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        (Tracer of) DeviceArray: u'(x).
    """

    def ux(X):
        return grad(net_u, argnums=1)(params, X)

    return jit(ux)


def net_uxx(params):
    """
    Defines neural network for second spatial derivative of u(x): u''(x).

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        (Tracer of) DeviceArray: u''(x).
    """

    def uxx(X):
        u_x = net_ux(params)
        return grad(u_x)(X)

    return jit(uxx)


@jit
def funx(X):
    """
    The f(x) in the partial derivative equation.

    Args:
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        (Tracer of) DeviceArray: Elementwise exponent of X.
    """
    return jnp.exp(X)


@jit
def loss_f(params, X, nu):
    """
    Calculates our residual loss.

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        nu (Tracer of float): Multiplicative constant.

    Returns:
        (Tracer of) DeviceArray: Residual loss.
    """
    u = vmap(net_u, (None, 0))(params, X)
    u_xxf = net_uxx(params)
    u_xx = vmap(u_xxf, (0))(X)
    fx = vmap(funx, (0))(X)
    res = nu * u_xx - u - fx
    loss_f = jnp.mean((res.flatten()) ** 2)
    return loss_f


@jit
def loss_b(params):
    """
    Calculates the boundary loss.

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.

    Returns:
        (Tracer of) DeviceArray: Boundary loss.
    """

    loss_b = (net_u(params, -1) - 1) ** 2 + (net_u(params, 1)) ** 2
    return loss_b


@jit
def loss(params, X, nu):
    """
    Combines the boundary loss and residue loss into a single loss matrix.

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        nu (Tracer of float): Multiplicative constant.

    Returns:
        (Tracer of) DeviceArray: Total loss matrix.
    """
    lossf = loss_f(params, X, nu)
    lossb = loss_b(params)
    return lossb + lossf


@jit
def step(istep, opt_state, X):
    """
    Training step that computes gradients for network weights and applies the Adam
    optimizer to the network.

    Args:
        istep (int): Current iteration step number.
        opt_state (Tracer of OptimizerState): Optimised network parameters.
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        (Tracer of) DeviceArray: Optimised network parameters.
    """
    param = get_params(opt_state)
    g = grad(loss, argnums=0)(param, X, nu)
    return opt_update(istep, g, opt_state)


#######################################################
###              MODEL HYPERPARAMETERS              ###
#######################################################

"""
Model Hyperparameter initalisation.

Defined hyperparameters: 
    nu (float): Multiplicative constant.
    layer_sizes (list[int]): Network architecture.
    nIter (int): Number of epochs / iterations.
"""

nu = 10 ** (-3)
layer_sizes = [1, 20, 20, 20, 1]
nIter = 20000 + 1

"""
Initialising weights, biases.

Weights and Biases:
    params (list[DeviceArray[float]]): Initialised weights and biases.
"""

params = init_network_params(layer_sizes, random.PRNGKey(0))

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

# Generation of 'input data', known as collocation points.
x = jnp.arange(-1, 1.05, 0.05)

#######################################################
###                  MODEL TRAINING                 ###
#######################################################

pbar = trange(nIter)
for it in pbar:
    opt_state = step(it, opt_state, x)
    if it % 1 == 0:
        params = get_params(opt_state)
        l_b = loss_b(params)
        l_f = loss_f(params, x, nu)
        pbar.set_postfix({"Loss_res": l_f, "loss_bound": l_b})
        lb_list.append(l_b)
        lf_list.append(l_f)

# final prediction of u(x)
u_pred = vmap(predict, (None, 0))(params, x)

#######################################################
###                     PLOTTING                    ###
#######################################################

fig, axs = plt.subplots(1, 2)

axs[0].plot(x, u_pred)
axs[0].set_title("Simple PINN Proposed Solution")
axs[0].set_xlabel("x")
axs[0].set_ylabel("Predicted u(x)")

axs[1].plot(lb_list, label="Boundary loss")
axs[1].plot(lf_list, label="Residue loss")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Loss")
axs[1].legend()
axs[1].set_title("Residue and Function Loss vs. Epochs")

plt.show()
