import numpy as np

def random_initialization(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-0.01, 0.01, size=shape)

def xavier_initialization(shape, alpha=1.0, seed=None):
    if len(shape) < 2:
        raise ValueError("Xavier initialization requires at least 2D shape")
    n_in = shape[0]
    if len(shape) > 1:
        n_out = shape[1]
    else:
        n_out = 1
    limit = np.sqrt(6.0 / (n_in + n_out))
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-limit, limit, size=shape)

def he_initialization(shape, alpha=2.0, seed=None):
    if len(shape) < 2:
        raise ValueError("He initialization requires at least 2D shape")
    n_in = shape[0]
    std = np.sqrt(2.0 / n_in)
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(0.0, std, size=shape)

def zeros_initialization(shape):
    return np.zeros(shape)

INITIALIZERS = {'random': random_initialization, 'xavier': xavier_initialization, 'he': he_initialization, 'zeros': zeros_initialization}

def get_initializer(name):
    if name not in INITIALIZERS:
        raise ValueError(f"Unknown initializer: {name}")
    return INITIALIZERS[name]

def initialize_weights(input_size, output_size, method='xavier', activation='relu', seed=None):
    initializer = get_initializer(method)
    weight_shape = (input_size, output_size)
    if method == 'zeros':
        weights = initializer(weight_shape)
    else:
        weights = initializer(weight_shape, seed=seed)
    bias_shape = (output_size,)
    biases = zeros_initialization(bias_shape)
    return weights, biases
