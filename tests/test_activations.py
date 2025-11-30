# Unit tests for activation functions

import sys
import os
import numpy as np
import pytest
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from activations import (
    relu, relu_derivative,
    sigmoid, sigmoid_derivative,
    tanh, tanh_derivative,
    softmax, softmax_derivative,
    get_activation, get_activation_derivative
)

class TestReLU:
    # Test ReLU activation and derivative
    
    def test_relu_positive(self):
        # Test ReLU with positive values
        x = np.array([1.0, 2.0, 3.0])
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu(x), expected)
    
    def test_relu_negative(self):
        # Test ReLU with negative values
        x = np.array([-1.0, -2.0, -3.0])
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(relu(x), expected)
    
    def test_relu_mixed(self):
        # Test ReLU with mixed values
        x = np.array([-1.0, 0.0, 1.0])
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(relu(x), expected)
    
    def test_relu_derivative(self):
        # Test ReLU derivative
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(relu_derivative(x), expected)


class TestSigmoid:
    # Test Sigmoid activation and derivative
    
    def test_sigmoid_zero(self):
        # Test sigmoid at zero (should be 0.5)
        x = np.array([0.0])
        result = sigmoid(x)
        assert np.isclose(result[0], 0.5)
    
    def test_sigmoid_range(self):
        # Test sigmoid output is between 0 and 1
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = sigmoid(x)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_sigmoid_symmetry(self):
        # Test sigmoid symmetry: sigmoid(-x) = 1 - sigmoid(x)
        x = np.array([1.0, 2.0, 3.0])
        assert np.allclose(sigmoid(-x), 1 - sigmoid(x))
    
    def test_sigmoid_derivative_range(self):
        # Test sigmoid derivative is non-negative (max at x=0 is 0.25)
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = sigmoid_derivative(x)
        assert np.all(result >= 0)
        assert np.all(result <= 0.25)  # Max at x=0 is 0.25


class TestTanh:
    # Test Tanh activation and derivative
    
    def test_tanh_zero(self):
        # Test tanh at zero (should be 0)
        x = np.array([0.0])
        result = tanh(x)
        assert np.isclose(result[0], 0.0)
    
    def test_tanh_range(self):
        # Test tanh output is between -1 and 1
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = tanh(x)
        assert np.all(result >= -1)
        assert np.all(result <= 1)
    
    def test_tanh_symmetry(self):
        # Test tanh is odd: tanh(-x) = -tanh(x)
        x = np.array([1.0, 2.0, 3.0])
        assert np.allclose(tanh(-x), -tanh(x))
    
    def test_tanh_derivative_range(self):
        # Test tanh derivative is non-negative (max at x=0 is 1)
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = tanh_derivative(x)
        assert np.all(result >= 0)
        assert np.all(result <= 1)  # Max at x=0 is 1

class TestSoftmax:
    # Test Softmax activation

    def test_softmax_sum_to_one(self):
        # Test softmax outputs sum to 1 (probability distribution)
        x = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
        result = softmax(x)
        row_sums = np.sum(result, axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_softmax_positive(self):
        # Test softmax outputs are positive
        x = np.array([[1.0, 2.0, 3.0]])
        result = softmax(x)
        assert np.all(result > 0)
    
    def test_softmax_large_values(self):
        # Test softmax with large values (numerical stability check)
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = softmax(x)
        assert np.isfinite(result).all()
        assert np.allclose(np.sum(result, axis=1), 1.0)
    
    def test_softmax_max_output(self):
        # Test softmax gives highest probability to largest input
        x = np.array([[1.0, 5.0, 2.0]])
        result = softmax(x)
        assert np.argmax(result[0]) == 1  # Index of 5.0


class TestGetActivation:
    # Test get_activation helper functions
    
    def test_get_activation_valid(self):
        # Test getting valid activation functions
        assert get_activation('relu') == relu
        assert get_activation('sigmoid') == sigmoid
        assert get_activation('tanh') == tanh
        assert get_activation('softmax') == softmax
    
    def test_get_activation_invalid(self):
        # Test getting invalid activation raises error
        with pytest.raises(ValueError):
            get_activation('invalid_activation')
    
    def test_get_derivative_valid(self):
        # Test getting valid activation derivatives
        assert get_activation_derivative('relu') == relu_derivative
        assert get_activation_derivative('sigmoid') == sigmoid_derivative
        assert get_activation_derivative('tanh') == tanh_derivative
    
    def test_get_derivative_invalid(self):
        # Test getting invalid derivative raises error
        with pytest.raises(ValueError):
            get_activation_derivative('invalid_activation')


class TestActivationShapes:
    # Test activation functions preserve shapes
    
    def test_shape_preservation(self):
        # Test all activations preserve input shape
        shapes = [(5,), (3, 4), (2, 3, 4)]
        
        for shape in shapes:
            x = np.random.randn(*shape)
            
            # Test ReLU
            assert relu(x).shape == shape
            assert relu_derivative(x).shape == shape
            
            # Test sigmoid
            assert sigmoid(x).shape == shape
            assert sigmoid_derivative(x).shape == shape
            
            # Test tanh
            assert tanh(x).shape == shape
            assert tanh_derivative(x).shape == shape
    
    def test_softmax_2d_only(self):
        # Test softmax works with 2D arrays (batch_size, num_classes)
        x = np.random.randn(5, 10)
        result = softmax(x)
        assert result.shape == (5, 10)
        assert np.allclose(np.sum(result, axis=1), 1.0)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
