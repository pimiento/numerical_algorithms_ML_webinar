#!/usr/bin/env python3
from typing import List, Dict, Tuple
import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def predict(X: np.ndarray, y: np.ndarray, params: Dict[str, np.ndarray]):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    params -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    p = np.zeros((1,m))


    # Forward propagation
    A1, _ = linear_activation_forward(X, parameters["W1"], parameters["b1"], "relu")
    probas, _ = linear_activation_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        p[0,i] = 1 if probas[0,i] > 0.5 else 0
    print("Accuracy: "  + str(np.sum((p == y)/m)))

    return p

def print_mislabeled_images(
        clses: np.ndarray, X: np.ndarray, y: np.ndarray, p: np.ndarray
) -> None:
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " +
            clses[int(p[0,index])].decode("utf-8") +
            " \n Class: " +
            clses[y[0,index]].decode("utf-8")
        )



def sigmoid(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache


def sigmoid_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache

    s, _ = sigmoid(Z)
    dZ = dA * s * (1-s)

    assert dZ.shape == Z.shape

    return dZ


def relu(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    A = np.maximum(0,Z)

    assert A.shape == Z.shape

    cache = Z
    return A, cache


def relu_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert dZ.shape == Z.shape

    return dZ


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and test images from h5 file"""
    train_ds = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_y_orig = np.array(train_ds["train_set_y"][:])
    test_ds = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_y_orig = np.array(test_ds["test_set_y"][:])


    return (
        np.array(train_ds["train_set_x"][:]),
        train_set_y_orig.reshape((1, train_set_y_orig.shape[0])),
        np.array(test_ds["test_set_x"][:]),
        test_set_y_orig.reshape((1, test_set_y_orig.shape[0])),
        np.array(test_ds["list_classes"][:])
    )


def initialize_parameters(n_x: int, n_h: int, n_y: int) -> Dict[str, np.ndarray]:
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }


def linear_forward(
        A: np.ndarray, W: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, num of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ;
            stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(
        A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str
) -> Tuple[np.ndarray, Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data):
            (size of previous layer, num of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string:
            "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    assert activation in ("sigmoid", "relu")
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def compute_cost(AL: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, num of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat),
            shape (1, num of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = (-1/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
    cost = np.squeeze(cost)

    return cost


def linear_backward(
        dZ: np.ndarray,
        cache: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1),
            same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, _ = cache
    m = A_prev.shape[1]

    dW = (1/m) * (np.dot(dZ, A_prev.T))
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(
        dA: np.ndarray,
        cache: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        activation: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache)
            we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string:
            "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1),
            same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    assert activation in ("relu", "sigmoid")
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def update_parameters(
        params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], l_rate: float
) -> Dict[str, np.ndarray]:
    """
    Update parameters using gradient descent

    Arguments:
    params -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    _parameters = params.copy()
    L = len(_parameters) // 2 # num of layers in the neural network

    for l in range(L):
        _parameters[f"W{l+1}"] = params[f"W{l+1}"] - (l_rate * grads[f"dW{l+1}"])
        _parameters[f"b{l+1}"] = params[f"b{l+1}"] - (l_rate * grads[f"db{l+1}"])
    return _parameters

### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075


def two_layer_model(
        X: np.ndarray,
        Y: np.ndarray,
        layers_dims: Tuple[int, int, int],
        learning_rate: float = 0.0075,
        num_iterations: int = 3000,
        print_cost: bool = False
) -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    params = initialize_parameters(*layers_dims)

    # YOUR CODE ENDS HERE

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation:
        # LINEAR -> RELU -> LINEAR -> SIGMOID.
        # Inputs: "X, W1, b1, W2, b2".
        # Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        cost = compute_cost(A2, Y)

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation.
        # Inputs: "dA2, cache2, cache1".
        # Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        params = update_parameters(params, grads, learning_rate)
        # YOUR CODE ENDS HERE

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = params["W1"]
        b1 = params["b1"]
        W2 = params["W2"]
        b2 = params["b2"]

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return params, costs

def plot_costs(costs: List[np.ndarray], learning_rate: float = 0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title(f"Learning rate = {learning_rate}")
    plt.show()

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# The "-1" makes reshape flatten the remaining dimensions
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

### CONSTANTS DEFINING THE MODEL ####
N_X = num_px * num_px * 3
N_H = 7
N_Y = 1
LEARNING_RATE = 0.0075

parameters, costs = two_layer_model(
    train_x, train_y,
    layers_dims=(N_X, N_H, N_Y),
    num_iterations=2500,
    learning_rate=LEARNING_RATE,
    print_cost=True
)
# plot_costs(costs, learning_rate)

# pred_test = predict(test_x, test_y, parameters)
# print_mislabeled_images(classes, test_x, test_y, pred_test)
