import numpy as np
from .initializers import initialize_weights
from .layers import DenseLayer
from .activations import get_activation, get_activation_derivative
from .losses import get_loss_function, get_loss_derivative, l2_regularization
from .optimizers import get_optimizer

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', output_activation='softmax', 
                 learning_rate=0.001, optimizer='adam', weight_init='he', l2_lambda=0.0, dropout_rate=0.0, random_seed=None):
        self.layers = []
        input_dim = input_size
        for i in range(len(hidden_layers)):
            hidden_units = hidden_layers[i]
            new_layer = DenseLayer(input_size=input_dim, output_size=hidden_units, activation=activation, weight_init=weight_init, seed=random_seed)
            self.layers.append(new_layer)
            input_dim = hidden_units
        output_layer = DenseLayer(input_size=input_dim, output_size=output_size, activation=output_activation, weight_init=weight_init, seed=random_seed)
        self.layers.append(output_layer)
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.training = True
        self.optimizer = get_optimizer(optimizer, learning_rate=learning_rate)
        self.loss_function = 'cross_entropy'
        self.last_predictions = None
        self.last_loss = None
        self.dropout_masks = []
    
    def forward(self, X):
        A = X
        self.dropout_masks = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.activation_cache['A_prev'] = A
            Z = A @ layer.W + layer.b
            layer.activation_cache['Z'] = Z
            activation_fn = get_activation(layer.activation)
            A = activation_fn(Z)
            is_hidden_layer = (i < len(self.layers) - 1)
            if self.training and is_hidden_layer and self.dropout_rate > 0.0:
                random_values = np.random.random(A.shape)
                dropout_mask = (random_values > self.dropout_rate).astype(float)
                dropout_mask = dropout_mask / (1.0 - self.dropout_rate)
                A = A * dropout_mask
                self.dropout_masks.append(dropout_mask)
            else:
                self.dropout_masks.append(None)
            layer.activation_cache['A'] = A
        return A
    
    def backward(self, X, y, y_pred=None):
        if y_pred is None:
            y_pred = self.forward(X)
        loss_deriv_fn = get_loss_derivative(self.loss_function)
        dA = loss_deriv_fn(y_pred, y)
        m = X.shape[0]
        num_layers = len(self.layers)
        for idx in range(num_layers - 1, -1, -1):
            layer = self.layers[idx]
            Z = layer.activation_cache['Z']
            A_prev = layer.activation_cache['A_prev']
            is_output_layer = (idx == num_layers - 1)
            if is_output_layer and self.output_activation == 'softmax' and self.loss_function == 'cross_entropy':
                dZ = dA
            else:
                activation_grad_fn = get_activation_derivative(layer.activation)
                activation_grad = activation_grad_fn(Z)
                dZ = dA * activation_grad
            layer.dW = A_prev.T @ dZ
            layer.db = np.sum(dZ, axis=0, keepdims=True)
            if layer.b.shape != layer.db.shape:
                layer.db = layer.db.reshape(layer.b.shape)
            if self.l2_lambda > 0:
                l2_term = (self.l2_lambda / m) * layer.W
                layer.dW = layer.dW + l2_term
            dA = dZ @ layer.W.T
            if idx > 0:
                prev_layer_idx = idx - 1
                if prev_layer_idx < len(self.dropout_masks):
                    dropout_mask = self.dropout_masks[prev_layer_idx]
                    if dropout_mask is not None:
                        dA = dA * dropout_mask
    
    def update_weights(self):
        params = {}
        grads = {}
        for i in range(len(self.layers)):
            layer = self.layers[i]
            weight_key = 'W' + str(i+1)
            bias_key = 'b' + str(i+1)
            params[weight_key] = layer.W
            params[bias_key] = layer.b
            grads[weight_key] = layer.dW
            grads[bias_key] = layer.db
        updated_params = self.optimizer.update(params, grads)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            weight_key = 'W' + str(i+1)
            bias_key = 'b' + str(i+1)
            layer.W = updated_params[weight_key]
            layer.b = updated_params[bias_key]
    
    def compute_loss(self, y_pred, y_true):
        loss_func = get_loss_function(self.loss_function)
        data_loss = loss_func(y_pred, y_true)
        if self.l2_lambda > 0:
            weights_list = []
            for layer in self.layers:
                weights_list.append(layer.W)
            reg_loss = l2_regularization(weights_list, self.l2_lambda)
            total_loss = data_loss + reg_loss
        else:
            total_loss = data_loss
        return total_loss
    
    def train_step(self, X_batch, y_batch):
        y_pred = self.forward(X_batch)
        loss = self.compute_loss(y_pred, y_batch)
        self.backward(X_batch, y_batch, y_pred=y_pred)
        self.update_weights()
        self.last_predictions = y_pred
        self.last_loss = loss
        return loss
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def predict_proba(self, X):
        was_training = self.training
        self.training = False
        probabilities = self.forward(X)
        self.training = was_training
        return probabilities
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'W{i+1}'] = layer.W.copy()
            params[f'b{i+1}'] = layer.b.copy()
        return params
    
    def set_params(self, params):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            weight_key = 'W' + str(i+1)
            bias_key = 'b' + str(i+1)
            layer.W = params[weight_key].copy()
            layer.b = params[bias_key].copy()
