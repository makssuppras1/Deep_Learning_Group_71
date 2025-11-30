import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x):
    s = softmax(x)
    return s - np.square(s)

ACTIVATION_FUNCTIONS = {'relu': relu, 'sigmoid': sigmoid, 'tanh': tanh, 'softmax': softmax}
ACTIVATION_DERIVATIVES = {'relu': relu_derivative, 'sigmoid': sigmoid_derivative, 'tanh': tanh_derivative, 'softmax': softmax_derivative}

def get_activation(name):
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_FUNCTIONS[name]

def get_activation_derivative(name):
    if name not in ACTIVATION_DERIVATIVES:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_DERIVATIVES[name]
