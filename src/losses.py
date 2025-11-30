import numpy as np

def mean_squared_error(y_pred, y_true):
    return float(np.mean(np.square(y_pred - y_true)))

def mse_derivative(y_pred, y_true):
    return (2 / y_pred.size) * (y_pred - y_true)

def cross_entropy_loss(y_pred, y_true):
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    if y_pred_clipped.ndim == 1:
        y_pred_clipped = y_pred_clipped.reshape(1, -1)
        y_true = y_true.reshape(1, -1)
    return float(-np.sum(y_true * np.log(y_pred_clipped)) / y_pred_clipped.shape[0])

def cross_entropy_derivative(y_pred, y_true):
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
        y_true = y_true.reshape(1, -1)
    return (y_pred - y_true) / y_pred.shape[0]

def binary_cross_entropy(y_pred, y_true):
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)))

def l2_regularization(weights, lambda_):
    l2_sum = 0.0
    for W in weights:
        l2_sum = l2_sum + np.sum(np.square(W))
    L2 = (lambda_ / 2) * l2_sum
    return L2

LOSS_FUNCTIONS = {'mse': mean_squared_error, 'cross_entropy': cross_entropy_loss, 'binary_cross_entropy': binary_cross_entropy}
LOSS_DERIVATIVES = {'mse': mse_derivative, 'cross_entropy': cross_entropy_derivative, 'binary_cross_entropy': cross_entropy_derivative}

def get_loss_function(name):
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_FUNCTIONS[name]

def get_loss_derivative(name):
    if name not in LOSS_DERIVATIVES:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_DERIVATIVES[name]
