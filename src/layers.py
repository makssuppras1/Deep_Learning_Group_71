import numpy as np
from .initializers import initialize_weights

class DenseLayer:
    def __init__(self, input_size, output_size, activation='relu', weight_init='xavier', seed=None):
        self.W, self.b = initialize_weights(input_size, output_size, method=weight_init, seed=seed)
        self.activation = activation
        self.activation_cache = {'A_prev': None, 'Z': None, 'A': None}
        self.dW = None
        self.db = None
