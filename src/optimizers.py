import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        raise NotImplementedError

class SGD(Optimizer):
    def update(self, params, grads):
        updated_params = {}
        for key in params:
            if key in grads:
                updated_params[key] = params[key] - self.learning_rate * grads[key]
        return updated_params

class MomentumSGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params, grads):
        updated_params = {}
        for key in params:
            if key not in grads:
                raise ValueError(f"Gradient for parameter '{key}' not found")
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            updated_params[key] = params[key] + self.velocity[key]
        return updated_params

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, params, grads):
        updated_params = {}
        for key in params:
            if key not in grads:
                raise ValueError(f"Gradient for parameter '{key}' not found")
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            self.cache[key] = self.decay_rate * self.cache[key] + (1 - self.decay_rate) * grads[key] ** 2
            updated_params[key] = params[key] - self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        return updated_params

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def update(self, params, grads):
        self.t += 1
        updated_params = {}
        for key in params:
            if key not in grads:
                raise ValueError(f"Gradient for parameter '{key}' not found")
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
            if key not in self.v:
                self.v[key] = np.zeros_like(params[key])
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated_params

OPTIMIZERS = {'sgd': SGD, 'momentum': MomentumSGD, 'rmsprop': RMSprop, 'adam': Adam}

def get_optimizer(name, **kwargs):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    return OPTIMIZERS[name](**kwargs)
