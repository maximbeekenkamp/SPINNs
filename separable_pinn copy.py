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
A Separable PINN is a partial derivative solver that improves on a simple PINN model by leveraging forward-mode
autodifferentiation and operating on a per-axis basis. 

This 2D time dependent implementation of a PINN solves a two-dimensional differential heat equation defined by: 

-Delta u() = (4*10^6 * x^2 - 2*10^6 * x + 4*10^6 * y^2 - 2*10^6 * y + 49600) * exp (-1000[(x_1 - r_c)^2 + (x_2 - r_c)^2]) 
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
        key (DeviceArray): Key for random modules.
        scale (DeviceArray[float]): Float value between 0 and 1 to scale the initialisation.

    Returns:
        Tuple[DeviceArray]: Randomised initialisation for a single layer.
    """
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (m, n)), jnp.zeros(n)


def init_network_params(sizes, key):
    """
    Initializes all layers for a fully-connected neural network with
    size "sizes".

    Args:
        sizes (List[int]): Network architecture.
        key (DeviceArray): Key for random modules.

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
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        (Tracer of) DeviceArray: Output predictions (u_pred).
    """
    activations = X
    for w, b in params[:-1]:
        activations = tanh(jnp.dot(activations, w) + b)
    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits
#does this sum work if it is outputting multiple things
#should each input get r outputs then the first outputs for each x input will get 
#tensor producted with the first outputs for each y input and so on then sum r tensor products?


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

def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jvp(f, (primals,), tangents)[1]
    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out

#take gradients before merging
@jit
def net_bigu(x_params, y_params, X, Y):
    u_x = vmap(net_u, (None, 0))(x_params, X)
    u_y = vmap(net_u, (None, 0))(y_params, Y)
    return jnp.sum(jnp.einsum('in,jn->nij', u_x, u_y), axis=0)

@jit
def net_laplace(x_params, y_params, X, Y):
    v = jnp.ones(x.shape)
    u_xx = hvp_fwdfwd(lambda x: net_bigu(x_params, y_params, x, y), (x,), (v,))
    u_yy = hvp_fwdfwd(lambda y: net_bigu(x_params, y_params, x, y), (y,), (v,))
    return u_xx + u_yy

@jit
def funxy(X, Y):
    """
    The f(x) in the partial derivative equation.

    Args:
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        (Tracer of) DeviceArray: Elementwise exponent of X.
    """
    return (4*10**6 * X**2 - 2*10**6 * X + 4*10**6 * Y**2 - 2*10**6 * Y + 49600) * jnp.exp(-1000*((X - 0.5)**2 + (Y - 0.5)**2))

@jit
def finalfunc(X,Y):
    return jnp.exp(-1000*((X - 0.5)**2 + (Y - 0.5)**2))


@jit
def loss(x_params, y_params, X, Y, bound, bfilter):
    """
    Calculates our residual loss.

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        nu (Tracer of float): Multiplicative constant.

    Returns:
        (Tracer of) DeviceArray: Residual loss.
    """
    u_laplace = net_laplace(x_params, y_params, X, Y)
    fxy = vmap(vmap(funxy, in_axes=(None,0)), in_axes=(0, None))(X, Y)
    res = u_laplace + fxy
    lossb = loss_b(u_laplace, bound, bfilter)
    lossf = jnp.mean((res.flatten())**2)
    loss = jnp.sum(lossf + lossb)
    return (loss, (lossf, lossb))


@jit
def loss_b(values, bound, bfilter):
    """
    Calculates the boundary loss.

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.

    Returns:
        (Tracer of) DeviceArray: Boundary loss.
    """

    return jnp.sum((values * bfilter - bound).flatten()**2)/(2*len(values[0]) + 2*len(values) - 4)

@jit
def step(istep, opt_state_x, opt_state_y, X, Y, bound, bfilter):
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
    param_x = get_params_x(opt_state_x)
    param_y = get_params_y(opt_state_y)
    g_x = grad(loss, argnums=0, has_aux=True)(param_x, param_y, X, Y, bound, bfilter)
    g_x = g_x[0]
    g_y = grad(loss, argnums=1, has_aux=True)(param_x, param_y, X, Y, bound, bfilter)
    g_y = g_y[0]
    return opt_update_x(istep, g_x, opt_state_x), opt_update_y(istep, g_y, opt_state_y)

def setup_boundry(X,Y):
    bound = np.array([[0 for _ in range(len(X))] for _ in range(len(Y))])
    bfilter = np.array([[0 for _ in range(len(X) - 2)] for _ in range(len(Y) - 2)])
    bfilter = np.pad(bfilter, ((1,1), (1,1)), constant_values = 1)
    for y in range(len(Y)):
        bound[y][0] = finalfunc(X[0],Y[y])
        bound[y][len(X)-1] = finalfunc(X[len(X)-1], Y[y])
    for x in range(len(X)):
        bound[0][x] = finalfunc(X[x],Y[0])
        bound[len(Y)-1][x] = finalfunc(X[x], Y[len(Y)-1])
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
r = 64
nu = 10 ** (-3)
layer_sizes = [1, 64, 64, 64, r]
nIter = 10000 + 1

"""
Initialising weights, biases.

Weights and Biases:
    params (list[DeviceArray[float]]): Initialised weights and biases.
"""

params_x = init_network_params(layer_sizes, random.PRNGKey(0))
params_y = init_network_params(layer_sizes, random.PRNGKey(0))

"""
Initialising optimiser for weights/biases.

Optimiser:
    opt_state (list[DeviceArray[float]]): Initialised optimised weights and biases state.
"""

opt_init_x, opt_update_x, get_params_x = optimizers.adam(5e-4)
opt_state_x = opt_init_x(params_x)

opt_init_y, opt_update_y, get_params_y = optimizers.adam(5e-4)
opt_state_y = opt_init_y(params_y)

# lists for boundary and residual loss values during training.
lb_list = []
lf_list = []


#######################################################
###                  MODEL TRAINING                 ###
#######################################################
bound, bfilter = setup_boundry(x,y)
pbar = trange(nIter)

start = time.time()
for it in pbar:
    opt_state_x, opt_state_y = step(it, opt_state_x, opt_state_y, x, y, bound, bfilter)
    if it % 1 == 0:
        params_x = get_params_x(opt_state_x)
        params_y = get_params_y(opt_state_y)
        loss_full, losses = loss(params_x, params_y, x, y, bound, bfilter)
        l_b = int(losses[1])
        l_f = int(losses[0])

        pbar.set_postfix({"Loss": (loss_full, losses)})
        lb_list.append(l_b)
        lf_list.append(l_f)

end = time.time()
print(f'Runtime: {((end-start)/nIter*1000):.2f} ms/iter.')

u_pred = net_bigu(params_x, params_y, x, y)

#######################################################
###                     PLOTTING                    ###
#######################################################
fig, axs = plt.subplots(1,2,figsize = (12,8))

shw = axs[0].imshow(u_pred, cmap='ocean')
axs[0].set_title("SPINN Proposed Solution")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
fig.colorbar(shw)

axs[1].plot(lb_list, label="Boundary loss")
axs[1].plot(lf_list, label="Residue loss")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Loss")
axs[1].legend()
axs[1].set_title("Loss vs. Epochs")

plt.show()
