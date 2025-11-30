import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_network import NeuralNetwork

def test_xor():
    model = NeuralNetwork(
        input_size=2,
        hidden_layers=[4],
        output_size=2,
        activation='relu',
        output_activation='softmax',
        learning_rate=0.1,
        optimizer='sgd'
    )
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    
    for i in range(100):
        loss = model.train_step(X, y)
    
    predictions = model.predict(X)
    expected = np.array([0, 1, 1, 0])
    matches = (predictions == expected)
    accuracy = np.mean(matches)
    
    return accuracy >= 0.75

def test_forward_shape():
    model = NeuralNetwork(
        input_size=10,
        hidden_layers=[20, 15],
        output_size=5,
        activation='relu',
        output_activation='softmax'
    )
    
    X = np.random.randn(32, 10)
    y_pred = model.forward(X)
    
    assert y_pred.shape == (32, 5)
    return True

def test_predict():
    model = NeuralNetwork(
        input_size=5,
        hidden_layers=[10],
        output_size=3,
        activation='relu',
        output_activation='softmax'
    )
    
    X = np.random.randn(10, 5)
    
    proba = model.predict_proba(X)
    assert proba.shape == (10, 3)
    proba_sums = np.sum(proba, axis=1)
    assert np.allclose(proba_sums, 1.0)
    
    predictions = model.predict(X)
    assert predictions.shape == (10,)
    assert np.all(predictions >= 0)
    assert np.all(predictions < 3)
    return True

if __name__ == '__main__':
    test_forward_shape()
    test_predict()
    test_xor()

