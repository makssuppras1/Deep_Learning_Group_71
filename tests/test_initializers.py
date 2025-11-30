# Tests for weight initialization methods (variance and distribution checks)
import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from initializers import (
    random_initialization,
    xavier_initialization,
    he_initialization,
    zeros_initialization,
    initialize_weights,
    get_initializer
)


class TestRandomInitialization:
    # Test random initialization
    
    def test_random_initialization_shape(self):
        # Test random initialization returns correct shape
        shape = (10, 20)
        weights = random_initialization(shape, seed=42)
        
        assert weights.shape == shape, \
            f"Expected shape {shape}, got {weights.shape}"
    
    def test_random_initialization_range(self):
        # Test random initialization values are in expected range [-0.01, 0.01]
        shape = (100, 200)
        weights = random_initialization(shape, seed=42)
        
        # Should be in range [-0.01, 0.01]
        assert np.all(weights >= -0.01), "Weights should be >= -0.01"
        assert np.all(weights <= 0.01), "Weights should be <= 0.01"
    
    def test_random_initialization_reproducibility(self):
        # Test random initialization is reproducible with same seed
        shape = (10, 20)
        weights1 = random_initialization(shape, seed=42)
        weights2 = random_initialization(shape, seed=42)
        
        np.testing.assert_array_equal(weights1, weights2), \
            "Same seed should produce same weights"


class TestXavierInitialization:
    # Test Xavier/Glorot initialization
    
    def test_xavier_initialization_shape(self):
        # Test Xavier initialization returns correct shape
        shape = (10, 20)
        weights = xavier_initialization(shape, seed=42)
        
        assert weights.shape == shape, \
            f"Expected shape {shape}, got {weights.shape}"
    
    def test_xavier_initialization_variance(self):
        # Test Xavier initialization has correct variance (2/(n_in+n_out) for uniform)
        n_in, n_out = 100, 50
        shape = (n_in, n_out)
        
        weights_list = []
        for seed in range(100):
            weights = xavier_initialization(shape, seed=seed)
            weights_flat = weights.flatten()
            weights_list.append(weights_flat)
        
        all_weights = np.concatenate(weights_list)
        sample_variance = np.var(all_weights)
        
        expected_variance = 2.0 / (n_in + n_out)
        
        assert np.isclose(sample_variance, expected_variance, rtol=0.2)
    
    def test_xavier_initialization_range(self):
        # Test Xavier initialization values are in expected range
        n_in, n_out = 10, 20
        shape = (n_in, n_out)
        weights = xavier_initialization(shape, seed=42)
        
        expected_range = np.sqrt(6.0 / (n_in + n_out))
        
        assert np.all(weights >= -expected_range * 1.1)
        assert np.all(weights <= expected_range * 1.1)
    
    def test_xavier_initialization_reproducibility(self):
        # Test Xavier initialization is reproducible with same seed
        shape = (10, 20)
        weights1 = xavier_initialization(shape, seed=42)
        weights2 = xavier_initialization(shape, seed=42)
        
        np.testing.assert_array_equal(weights1, weights2), \
            "Same seed should produce same weights"


class TestHeInitialization:
    # Test He initialization
    
    def test_he_initialization_shape(self):
        # Test He initialization returns correct shape
        shape = (10, 20)
        weights = he_initialization(shape, seed=42)
        
        assert weights.shape == shape, \
            f"Expected shape {shape}, got {weights.shape}"
    
    def test_he_initialization_variance(self):
        # Test He initialization has correct variance (2/n_in)
        n_in, n_out = 100, 50
        shape = (n_in, n_out)
        
        weights_list = []
        for seed in range(100):
            weights = he_initialization(shape, seed=seed)
            weights_flat = weights.flatten()
            weights_list.append(weights_flat)
        
        all_weights = np.concatenate(weights_list)
        sample_variance = np.var(all_weights)
        
        expected_variance = 2.0 / n_in
        
        assert np.isclose(sample_variance, expected_variance, rtol=0.2)
    
    def test_he_initialization_mean(self):
        # Test He initialization has zero mean
        n_in, n_out = 100, 50
        shape = (n_in, n_out)
        
        weights_list = []
        for seed in range(100):
            weights = he_initialization(shape, seed=seed)
            weights_flat = weights.flatten()
            weights_list.append(weights_flat)
        
        all_weights = np.concatenate(weights_list)
        sample_mean = np.mean(all_weights)
        
        assert np.isclose(sample_mean, 0.0, atol=0.01)
    
    def test_he_initialization_reproducibility(self):
        # Test He initialization is reproducible with same seed
        shape = (10, 20)
        weights1 = he_initialization(shape, seed=42)
        weights2 = he_initialization(shape, seed=42)
        
        np.testing.assert_array_equal(weights1, weights2), \
            "Same seed should produce same weights"


class TestZerosInitialization:
    # Test zeros initialization
    
    def test_zeros_initialization_shape(self):
        # Test zeros initialization returns correct shape
        shape = (10, 20)
        weights = zeros_initialization(shape)
        
        assert weights.shape == shape, \
            f"Expected shape {shape}, got {weights.shape}"
    
    def test_zeros_initialization_values(self):
        # Test zeros initialization returns all zeros
        shape = (10, 20)
        weights = zeros_initialization(shape)
        
        assert np.all(weights == 0), \
            "Zeros initialization should return all zeros"


class TestInitializeWeights:
    # Test initialize_weights helper function
    
    def test_initialize_weights_returns_tuple(self):
        # Test initialize_weights returns weights and biases tuple
        weights, biases = initialize_weights(
            input_size=10,
            output_size=20,
            method='xavier',
            seed=42
        )
        
        assert weights.shape == (10, 20), \
            f"Expected weights shape (10, 20), got {weights.shape}"
        assert biases.shape == (20,), \
            f"Expected biases shape (20,), got {biases.shape}"
    
    def test_initialize_weights_different_methods(self):
        # Test initialize_weights works with different methods (random, xavier, he, zeros)
        methods = ['random', 'xavier', 'he', 'zeros']
        
        for method in methods:
            weights, biases = initialize_weights(
                input_size=10,
                output_size=20,
                method=method,
                seed=42
            )
            
            assert weights.shape == (10, 20)
            assert biases.shape == (20,)


class TestGetInitializer:
    # Test get_initializer helper function
    
    def test_get_initializer_valid(self):
        # Test getting valid initializers
        assert get_initializer('random') == random_initialization
        assert get_initializer('xavier') == xavier_initialization
        assert get_initializer('he') == he_initialization
        assert get_initializer('zeros') == zeros_initialization
    
    def test_get_initializer_invalid(self):
        # Test getting invalid initializer raises error
        with pytest.raises(ValueError):
            get_initializer('invalid_initializer')


class TestInitializationVarianceScaling:
    # Test that initialization variance scales correctly with input size
    
    def test_xavier_variance_scales_with_input_size(self):
        # Test Xavier variance decreases with input size
        n_out = 50
        
        # Small input size
        n_in_small = 10
        shape_small = (n_in_small, n_out)
        weights_small = xavier_initialization(shape_small, seed=42)
        var_small = np.var(weights_small)
        
        # Large input size
        n_in_large = 100
        shape_large = (n_in_large, n_out)
        weights_large = xavier_initialization(shape_large, seed=42)
        var_large = np.var(weights_large)
        
        assert var_small > var_large
    
    def test_he_variance_scales_with_input_size(self):
        # Test He variance decreases with input size
        n_out = 50
        
        # Small input size
        n_in_small = 10
        shape_small = (n_in_small, n_out)
        weights_small = he_initialization(shape_small, seed=42)
        var_small = np.var(weights_small)
        
        # Large input size
        n_in_large = 100
        shape_large = (n_in_large, n_out)
        weights_large = he_initialization(shape_large, seed=42)
        var_large = np.var(weights_large)
        
        assert var_small > var_large


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

