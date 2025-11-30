# Unit tests for optimizers

import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimizers import SGD, MomentumSGD, RMSprop, Adam, get_optimizer


class TestSGD:
    # Test SGD optimizer
    
    def test_sgd_basic_update(self):
        # Test basic SGD update
        optimizer = SGD(learning_rate=0.01)
        
        params = {
            'W1': np.array([[1.0, 2.0], [3.0, 4.0]]),
            'b1': np.array([0.5, 0.5])
        }
        
        grads = {
            'W1': np.array([[0.1, 0.2], [0.3, 0.4]]),
            'b1': np.array([0.05, 0.05])
        }
        
        updated_params = optimizer.update(params, grads)
        
        expected_W1 = params['W1'] - 0.01 * grads['W1']
        expected_b1 = params['b1'] - 0.01 * grads['b1']
        
        np.testing.assert_array_almost_equal(updated_params['W1'], expected_W1)
        np.testing.assert_array_almost_equal(updated_params['b1'], expected_b1)
    
    def test_sgd_multiple_layers(self):
        # Test SGD with multiple layers
        optimizer = SGD(learning_rate=0.1)
        
        params = {
            'W1': np.array([[1.0, 2.0]]),
            'b1': np.array([1.0]),
            'W2': np.array([[3.0], [4.0]]),
            'b2': np.array([2.0])
        }
        
        grads = {
            'W1': np.array([[0.1, 0.2]]),
            'b1': np.array([0.1]),
            'W2': np.array([[0.3], [0.4]]),
            'b2': np.array([0.2])
        }
        
        updated_params = optimizer.update(params, grads)
        
        assert 'W1' in updated_params
        assert 'b1' in updated_params
        assert 'W2' in updated_params
        assert 'b2' in updated_params
        
        expected_W1 = params['W1'] - 0.1 * grads['W1']
        np.testing.assert_array_almost_equal(updated_params['W1'], expected_W1)
    
    def test_sgd_missing_gradient(self):
        # Test SGD raises error when gradient is missing
        optimizer = SGD(learning_rate=0.01)
        
        params = {
            'W1': np.array([[1.0, 2.0]]),
            'b1': np.array([1.0])
        }
        
        grads = {
            'W1': np.array([[0.1, 0.2]])
            # Missing 'b1' gradient
        }
        
        with pytest.raises(ValueError, match="Gradient for parameter 'b1' not found"):
            optimizer.update(params, grads)
    
    def test_sgd_different_learning_rates(self):
        # Test SGD with different learning rates
        params = {
            'W': np.array([[1.0, 2.0]])
        }
        grads = {
            'W': np.array([[0.1, 0.2]])
        }
        
        # Test with learning rate 0.01
        optimizer1 = SGD(learning_rate=0.01)
        updated1 = optimizer1.update(params, grads)
        
        # Test with learning rate 0.1
        optimizer2 = SGD(learning_rate=0.1)
        updated2 = optimizer2.update(params, grads)
        
        diff1 = np.abs(updated1['W'] - params['W'])
        diff2 = np.abs(updated2['W'] - params['W'])
        
        assert np.all(diff2 > diff1)
    
    def test_sgd_zero_gradient(self):
        # Test SGD with zero gradients (parameters should remain unchanged)
        optimizer = SGD(learning_rate=0.01)
        
        params = {
            'W': np.array([[1.0, 2.0], [3.0, 4.0]])
        }
        
        grads = {
            'W': np.zeros_like(params['W'])
        }
        
        updated_params = optimizer.update(params, grads)
        
        np.testing.assert_array_equal(updated_params['W'], params['W'])
    
    def test_sgd_shape_preservation(self):
        # Test that SGD preserves array shapes
        optimizer = SGD(learning_rate=0.01)
        
        params = {
            'W1': np.random.randn(10, 20),
            'b1': np.random.randn(20),
            'W2': np.random.randn(20, 5),
            'b2': np.random.randn(5)
        }
        
        grads = {
            'W1': np.random.randn(10, 20),
            'b1': np.random.randn(20),
            'W2': np.random.randn(20, 5),
            'b2': np.random.randn(5)
        }
        
        updated_params = optimizer.update(params, grads)
        
        assert updated_params['W1'].shape == params['W1'].shape
        assert updated_params['b1'].shape == params['b1'].shape
        assert updated_params['W2'].shape == params['W2'].shape
        assert updated_params['b2'].shape == params['b2'].shape


class TestGetOptimizer:
    # Test get_optimizer helper function
    
    def test_get_sgd(self):
        # Test getting SGD optimizer
        optimizer = get_optimizer('sgd', learning_rate=0.01)
        assert isinstance(optimizer, SGD)
        assert optimizer.learning_rate == 0.01
    
    def test_get_invalid_optimizer(self):
        # Test getting invalid optimizer raises error
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_optimizer('invalid_optimizer')


class TestOptimizerSimpleFunction:
    # Test optimizers on a simple function minimization
    
    def test_sgd_minimize_quadratic(self):
        optimizer = SGD(learning_rate=0.1)
        
        params = {'x': np.array([0.0])}
        
        for i in range(100):
            x_val = params['x'][0]
            grad_val = 2 * (x_val - 5)
            grads = {'x': np.array([grad_val])}
            params = optimizer.update(params, grads)
        
        x_final = params['x'][0]
        assert np.abs(x_final - 5.0) < 0.1
    
    def test_sgd_minimize_2d_function(self):
        optimizer = SGD(learning_rate=0.1)
        
        params = {
            'x': np.array([3.0]),
            'y': np.array([4.0])
        }
        
        for i in range(50):
            x_val = params['x'][0]
            y_val = params['y'][0]
            
            grad_x = 2 * x_val
            grad_y = 2 * y_val
            grads = {
                'x': np.array([grad_x]),
                'y': np.array([grad_y])
            }
            
            params = optimizer.update(params, grads)
        
        x_final = params['x'][0]
        y_final = params['y'][0]
        assert np.abs(x_final) < 0.1
        assert np.abs(y_final) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

