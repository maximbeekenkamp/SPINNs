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


@jit
def net_u(params, X, Y):
    """
    [Description]

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        Y (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        (Tracer of) DeviceArray: u(x).
    """
    xy_array = jnp.array([X,Y])
    return predict(params, xy_array)

@jit
def net_u_grad(params, X, Y):
    """
    [Description]

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        Y (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        (Tracer of) DeviceArray: u(x).
    """
    x_array = jnp.array([X, Y])
    y_pred = predict(params, x_array)
    return y_pred[0]


def net_f(params):
    """
    Defines neural network for first spatial derivative of u(x): u'(x).

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.

    Returns:
        (Tracer of) Tuple[DeviceArray]: u'(x), u'(y).
    """

    def ux(X,Y):
        return grad(net_u_grad, argnums=1)(params, X, Y)
    
    def uy(X,Y):
        return grad(net_u_grad, argnums=2)(params, X, Y)

    return jit(ux), jit(uy)


def net_ff(params):
    """
    Defines neural network for second spatial derivative of u(x): u''(x).

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        (Tracer of) DeviceArray: u''(x).
    """

    def uxx(X, Y):
        u_x, _ = net_f(params)
        return grad(u_x, argnums=0)(X, Y)
    
    def uyy(X,Y):
        _, u_y = net_f(params)
        return grad(u_y, argnums=1)(X, Y)

    return jit(uxx), jit(uyy)

#take gradients before merging
@jit
def net_bigu(params, X, Y):
    u_xxf, u_yyf = net_ff(params)
    u_xx = vmap(vmap(u_xxf, in_axes=(None, 0)), in_axes=(0, None))(X, Y)
    u_yy = vmap(vmap(u_yyf, in_axes=(None, 0)), in_axes=(0, None))(X, Y)
    laplace = u_xx + u_yy
    return laplace


@jit
def funxy(X, Y):
    """
    The f(x) in the partial derivative equation.

    Args:
        X (Tracer of DeviceArray): Collocation points in the domain.

    Returns:
        (Tracer of) DeviceArray: Elementwise exponent of X.
    """
    return (4*10**6 * X**2 + -4*10**6 * X + 4*10**6 * Y**2 - 4*10**6 * Y + 1.996*10**6) * jnp.exp(-1000*((X**2 - X + Y**2 - Y + 0.5)))

@jit
def finalfunc(X,Y):
    return jnp.exp(-1000*((X - 0.5)**2 + (Y - 0.5)**2))


@jit
def loss(params, X, Y, bound, bfilter):
    """
    Calculates our residual loss.

    Args:
        params (Tracer of list[DeviceArray]): List containing weights and biases.
        X (Tracer of DeviceArray): Collocation points in the domain.
        nu (Tracer of float): Multiplicative constant.

    Returns:
        (Tracer of) DeviceArray: Residual loss.
    """
    u = vmap(vmap(net_u, in_axes=(None,None,0)), in_axes=(None, 0, None))(params, X, Y)
    laplace = net_bigu(params, X, Y)
    fxy = vmap(vmap(funxy, in_axes=(None,0)), in_axes=(0, None))(X, Y)
    res = laplace - fxy
    lossb = loss_b(u, bound, bfilter)
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
def step(istep, opt_state, X, Y, bound, bfilter):
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
    params = get_params(opt_state)
    g = grad(loss, argnums=0, has_aux=True)(params, X, Y, bound, bfilter)
    return opt_update(istep, g[0], opt_state)

def setup_boundry(X,Y):
    bound = np.array([[0 for a in range(len(X))] for b in range(len(Y))])
    bfilter = np.array([[0 for a in range(len(X) - 2)] for b in range(len(Y) - 2)])
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

params = init_network_params(layer_sizes, random.PRNGKey(0))
lam = random.uniform(random.PRNGKey(0), shape=[1])

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

#######################################################
###                  MODEL TRAINING                 ###
#######################################################
bound, bfilter = setup_boundry(x,y)
pbar = trange(nIter)

start = time.time()
for it in pbar:
    opt_state = step(it, opt_state, x, y, bound, bfilter)
    if it % 1 == 0:
        params = get_params(opt_state)
        loss_full, losses = loss(params, x, y, bound, bfilter)
        l_b = int(losses[1])
        l_f = int(losses[0])

        pbar.set_postfix({"Loss": (loss_full, losses)})
        lb_list.append(l_b)
        lf_list.append(l_f)

end = time.time()
print(f'Runtime: {((end-start)/nIter*1000):.2f} ms/iter.')

u_pred = vmap(vmap(net_u, in_axes=(None,None,0)), in_axes=(None, 0, None))(params, x, y)

# print("lb",lb_list)
# print('lf', lf_list)

#######################################################
###                     PLOTTING                    ###
#######################################################
fig, axs = plt.subplots(1,2,figsize = (12,8))

shw = axs[0].imshow(u_pred, cmap='ocean')
axs[0].set_title("PINN Proposed Solution")
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
