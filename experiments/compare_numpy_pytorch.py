import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.pytorch_neural_network import PyTorchNeuralNetwork
from src.data_loader import load_cifar10, preprocess_data, create_mini_batches
from src.utils import set_random_seed


def compare_arrays(arr1, arr2, name="Arrays", rtol=1e-4, atol=1e-5):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    if arr1.shape != arr2.shape:
        print(f"x {name}: Shape mismatch! {arr1.shape} vs {arr2.shape}")
        return False
    
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    arr2_safe = arr2 + 1e-10
    relative_diff = np.max(np.abs(diff / arr2_safe))
    
    is_close = np.allclose(arr1, arr2, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"✔ {name}: Match! (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={relative_diff:.2e})")
    else:
        print(f"x {name}: Mismatch! (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={relative_diff:.2e})")
        arr1_flat = arr1.flatten()
        arr2_flat = arr2.flatten()
        arr1_first = arr1_flat[:5]
        arr2_first = arr2_flat[:5]
        print(f"   First few values - NumPy: {arr1_first}, PyTorch: {arr2_first}")
    
    return is_close


def copy_weights_from_numpy_to_pytorch(numpy_model, pytorch_model):
    pytorch_model.set_params(numpy_model.get_params())


def test_forward_pass(numpy_model, pytorch_model, X_test):
    print("\n" + "="*60)
    print("TEST 1: Forward Pass Outputs")
    print("="*60)
    numpy_model.eval()
    y_pred_numpy = numpy_model.predict_proba(X_test)
    X_torch = torch.from_numpy(X_test).float()
    y_pred_pytorch = pytorch_model.predict_proba(X_torch).cpu().numpy()
    return compare_arrays(y_pred_numpy, y_pred_pytorch, "Forward Pass Outputs")


def test_loss_computation(numpy_model, pytorch_model, X_test, y_test):
    print("\n" + "="*60)
    print("TEST 2: Loss Computation")
    print("="*60)
    numpy_model.eval()
    y_pred_numpy = numpy_model.predict_proba(X_test)
    loss_numpy = numpy_model.compute_loss(y_pred_numpy, y_test)
    X_torch = torch.from_numpy(X_test).float()
    y_torch = torch.from_numpy(y_test).float()
    y_pred_pytorch = pytorch_model.predict_proba(X_torch)
    loss_pytorch = pytorch_model.compute_loss(y_pred_pytorch, y_torch).item()
    return compare_arrays(np.array([loss_numpy]), np.array([loss_pytorch]), "Loss Values", rtol=1e-3)


def test_gradients(numpy_model, pytorch_model, X_batch, y_batch):
    print("\n" + "="*60)
    print("TEST 3: Gradient Computation")
    print("="*60)
    numpy_model.train()
    y_pred_numpy = numpy_model.forward(X_batch)
    numpy_model.backward(X_batch, y_batch, y_pred=y_pred_numpy)
    pytorch_model.train()
    X_torch = torch.from_numpy(X_batch).float()
    y_torch = torch.from_numpy(y_batch).float()
    y_pred_pytorch = pytorch_model.forward(X_torch)
    loss = pytorch_model.compute_loss(y_pred_pytorch, y_torch)
    pytorch_model.optimizer.zero_grad()
    loss.backward()
    
    if numpy_model.l2_lambda > 0:
        m = X_batch.shape[0]
        for layer in pytorch_model.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.grad += (numpy_model.l2_lambda / m) * layer.weight
    
    pytorch_linear_layers = []
    for layer in pytorch_model.layers:
        if isinstance(layer, torch.nn.Linear):
            pytorch_linear_layers.append(layer)
    
    all_match = True
    for i in range(len(numpy_model.layers)):
        numpy_layer = numpy_model.layers[i]
        pytorch_layer = pytorch_linear_layers[i]
        dW_numpy = numpy_layer.dW
        db_numpy = numpy_layer.db
        dW_pytorch = pytorch_layer.weight.grad.T.detach().cpu().numpy()
        db_pytorch = pytorch_layer.bias.grad.detach().cpu().numpy()
        layer_num = i + 1
        match_w = compare_arrays(dW_numpy, dW_pytorch, "Layer " + str(layer_num) + " Weight Gradients (dW)", rtol=1e-3)
        match_b = compare_arrays(db_numpy, db_pytorch, "Layer " + str(layer_num) + " Bias Gradients (db)", rtol=1e-3)
        if not match_w:
            all_match = False
        if not match_b:
            all_match = False
    
    return all_match


def test_training_step(numpy_model, pytorch_model, X_batch, y_batch):
    print("\n" + "="*60)
    print("TEST 4: Training Step (Weight Updates)")
    print("="*60)
    numpy_params_before = numpy_model.get_params()
    pytorch_params_before = pytorch_model.get_params()
    pytorch_model.reset_optimizer_state()
    numpy_model.train()
    loss_numpy = numpy_model.train_step(X_batch, y_batch)
    pytorch_model.train()
    X_torch = torch.from_numpy(X_batch).float()
    y_torch = torch.from_numpy(y_batch).float()
    loss_pytorch = pytorch_model.train_step(X_torch, y_torch)
    numpy_params_after = numpy_model.get_params()
    pytorch_params_after = pytorch_model.get_params()
    
    all_match = True
    for key in numpy_params_before.keys():
        numpy_update = numpy_params_after[key] - numpy_params_before[key]
        pytorch_update = pytorch_params_after[key] - pytorch_params_before[key]
        match = compare_arrays(numpy_update, pytorch_update, f"Weight Update ({key})", rtol=1e-1, atol=1e-3)
        if not match:
            all_match = False
    
    return all_match


def test_multiple_training_steps(numpy_model, pytorch_model, X_train, y_train, num_steps=5, seed=42):
    print("\n" + "="*60)
    print("TEST 5: Multiple Training Steps")
    print("="*60)
    batches = create_mini_batches(X_train, y_train, batch_size=32, shuffle=False)
    all_match = True
    
    max_steps = num_steps
    if len(batches) < num_steps:
        max_steps = len(batches)
    
    for step in range(max_steps):
        X_batch, y_batch = batches[step]
        current_seed = seed + step
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        copy_weights_from_numpy_to_pytorch(numpy_model, pytorch_model)
        pytorch_model.reset_optimizer_state()
        numpy_model.train()
        loss_numpy = numpy_model.train_step(X_batch, y_batch)
        pytorch_model.train()
        X_torch = torch.from_numpy(X_batch).float()
        y_torch = torch.from_numpy(y_batch).float()
        loss_pytorch = pytorch_model.train_step(X_torch, y_torch)
        numpy_model.eval()
        y_pred_numpy = numpy_model.predict_proba(X_batch)
        pytorch_model.eval()
        y_pred_pytorch = pytorch_model.predict_proba(X_torch).cpu().numpy()
        step_name = "Step " + str(step+1) + " Outputs"
        match = compare_arrays(y_pred_numpy, y_pred_pytorch, step_name, rtol=1e-2)
        if not match:
            all_match = False
        
        numpy_params = numpy_model.get_params()
        pytorch_params = pytorch_model.get_params()
        for key in numpy_params.keys():
            weight_name = "Step " + str(step+1) + " Weights (" + key + ")"
            match_w = compare_arrays(numpy_params[key], pytorch_params[key], weight_name, rtol=1e-2, atol=1e-4)
            if not match_w:
                all_match = False
    
    return all_match


def main():
    seed = 42
    set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    data = load_cifar10(data_dir)
    X_train_full = data[0]
    y_train_full = data[1]
    X_test = data[2]
    y_test = data[3]
    X_train_subset = X_train_full[:1000]
    y_train_subset = y_train_full[:1000]
    X_test_subset = X_test[:100]
    y_test_subset = y_test[:100]
    
    X_train_subset, y_train_subset = preprocess_data(X_train_subset, y_train_subset, num_classes=10, flatten=True, normalize=True)
    X_test_subset, y_test_subset = preprocess_data(X_test_subset, y_test_subset, num_classes=10, flatten=True, normalize=True)
    
    config = {
        'input_size': 3072,
        'hidden_layers': [128, 64],
        'output_size': 10,
        'activation': 'relu',
        'output_activation': 'softmax',
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'weight_init': 'he',
        'l2_lambda': 0.0001,
        'dropout_rate': 0.0,
        'random_seed': seed
    }
    
    numpy_model = NeuralNetwork(**config)
    torch.manual_seed(seed)
    np.random.seed(seed)
    pytorch_model = PyTorchNeuralNetwork(**config)
    copy_weights_from_numpy_to_pytorch(numpy_model, pytorch_model)
    
    numpy_params = numpy_model.get_params()
    pytorch_params = pytorch_model.get_params()
    initial_weights_match = True
    for key in numpy_params.keys():
        key_name = "Initial " + key
        match = compare_arrays(numpy_params[key], pytorch_params[key], key_name, rtol=1e-5)
        if not match:
            initial_weights_match = False
    
    if not initial_weights_match:
        print("Warning: Initial weights don't match exactly after copying!")
    
    results = {}
    results['forward_pass'] = test_forward_pass(numpy_model, pytorch_model, X_test_subset)
    results['loss'] = test_loss_computation(numpy_model, pytorch_model, X_test_subset, y_test_subset)
    X_batch = X_train_subset[:32]
    y_batch = y_train_subset[:32]
    copy_weights_from_numpy_to_pytorch(numpy_model, pytorch_model)
    results['gradients'] = test_gradients(numpy_model, pytorch_model, X_batch, y_batch)
    copy_weights_from_numpy_to_pytorch(numpy_model, pytorch_model)
    results['training_step'] = test_training_step(numpy_model, pytorch_model, X_batch, y_batch)
    copy_weights_from_numpy_to_pytorch(numpy_model, pytorch_model)
    results['multiple_steps'] = test_multiple_training_steps(numpy_model, pytorch_model, X_train_subset, y_train_subset, num_steps=5, seed=seed)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name in results.keys():
        passed = results[test_name]
        if passed:
            status = "✔ PASSED"
        else:
            status = "x FAILED"
        print(test_name + ": " + status)
    
    all_passed = True
    for test_name in results.keys():
        if not results[test_name]:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("(y) All tests passed! Networks produce similar results.")
    else:
        print("!!  Some tests failed. Networks may have differences.")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    main()

