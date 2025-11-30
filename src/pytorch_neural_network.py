import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PyTorchNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', output_activation='softmax',
                 learning_rate=0.001, optimizer='adam', weight_init='he', l2_lambda=0.0, dropout_rate=0.0, random_seed=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        layers = []
        input_dim = input_size
        for i in range(len(hidden_layers)):
            hidden_units = hidden_layers[i]
            layers.append(nn.Linear(input_dim, hidden_units))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_units
        layers.append(nn.Linear(input_dim, output_size))
        self.layers = nn.ModuleList(layers)
        self._initialize_weights(weight_init, random_seed)
        self._setup_optimizer()
        self.loss_function = 'cross_entropy'
    
    def _initialize_weights(self, weight_init, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, nn.Linear):
                if weight_init == 'he':
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.zeros_(layer.bias)
                elif weight_init == 'xavier':
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif weight_init == 'random':
                    nn.init.uniform_(layer.weight, -0.01, 0.01)
                    nn.init.zeros_(layer.bias)
    
    def _setup_optimizer(self):
        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0)
        elif self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=0.0)
        elif self.optimizer_name == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=0.0)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def forward(self, X):
        A = X
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            if isinstance(layer, nn.Linear):
                A = layer(A)
                if self.activation == 'relu':
                    A = F.relu(A)
                elif self.activation == 'sigmoid':
                    A = torch.sigmoid(A)
                elif self.activation == 'tanh':
                    A = torch.tanh(A)
            elif isinstance(layer, nn.Dropout):
                A = layer(A)
        output_layer = self.layers[-1]
        Z = output_layer(A)
        if self.output_activation == 'softmax':
            A = F.softmax(Z, dim=1)
        elif self.output_activation == 'sigmoid':
            A = torch.sigmoid(Z)
        else:
            A = Z
        return A
    
    def compute_loss(self, y_pred, y_true):
        if self.loss_function == 'cross_entropy':
            eps = 1e-12
            y_pred_clipped = torch.clamp(y_pred, eps, 1.0 - eps)
            data_loss = -torch.sum(y_true * torch.log(y_pred_clipped)) / y_pred.shape[0]
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
        if self.l2_lambda > 0:
            l2_sum = 0.0
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    l2_sum = l2_sum + torch.sum(layer.weight ** 2)
            reg_loss = (self.l2_lambda / 2) * l2_sum
            total_loss = data_loss + reg_loss
        else:
            total_loss = data_loss
        return total_loss
    
    def train_step(self, X_batch, y_batch):
        self.train()
        y_pred = self.forward(X_batch)
        loss = self.compute_loss(y_pred, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        if self.l2_lambda > 0:
            m = X_batch.shape[0]
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.grad += (self.l2_lambda / m) * layer.weight
        self.optimizer.step()
        return loss.item()
    
    def predict(self, X):
        return torch.argmax(self.predict_proba(X), dim=1).cpu().numpy()
    
    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X)
    
    def get_params(self):
        params = {}
        layer_idx = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, nn.Linear):
                layer_idx += 1
                weight_key = 'W' + str(layer_idx)
                bias_key = 'b' + str(layer_idx)
                weight_numpy = layer.weight.detach().cpu().numpy()
                weight_numpy = weight_numpy.T
                params[weight_key] = weight_numpy.copy()
                bias_numpy = layer.bias.detach().cpu().numpy()
                params[bias_key] = bias_numpy.copy()
        return params
    
    def set_params(self, params):
        layer_idx = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, nn.Linear):
                layer_idx += 1
                weight_key = 'W' + str(layer_idx)
                bias_key = 'b' + str(layer_idx)
                if weight_key in params:
                    weight_numpy = params[weight_key]
                    weight_numpy = weight_numpy.T
                    layer.weight.data = torch.from_numpy(weight_numpy).float()
                if bias_key in params:
                    bias_numpy = params[bias_key]
                    layer.bias.data = torch.from_numpy(bias_numpy).float()
    
    def reset_optimizer_state(self):
        if len(self.optimizer.state) == 0:
            dummy_input = torch.zeros(1, self.input_size, requires_grad=False)
            dummy_output = self.forward(dummy_input)
            dummy_loss = dummy_output.sum()
            self.optimizer.zero_grad()
            dummy_loss.backward()
            self.optimizer.zero_grad()
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    if 'exp_avg' in state:
                        state['exp_avg'].zero_()
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'].zero_()
                    if 'step' in state:
                        state['step'] = torch.tensor(0, dtype=torch.int32, device=param.device)
