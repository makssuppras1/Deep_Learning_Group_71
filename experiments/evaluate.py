import numpy as np
import sys
import os
import argparse
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.data_loader import load_fashion_mnist, load_cifar10, preprocess_data, get_class_names
from src.utils import accuracy_score, plot_confusion_matrix, print_classification_report


def evaluate_model(model, X_test, y_test, dataset_name, save_dir='results/plots'):
    class_names = get_class_names(dataset_name)
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_test_indices = np.argmax(y_test, axis=1)
    else:
        y_test_indices = y_test
    accuracy = accuracy_score(y_pred, y_test_indices)
    loss = model.compute_loss(y_pred_proba, y_test)
    print(f"Test Acc: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Test Loss: {loss:.4f}")
    print_classification_report(y_test_indices, y_pred, class_names)
    os.makedirs(save_dir, exist_ok=True)
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test_indices, y_pred, class_names, save_path=cm_path)
    return accuracy, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='results/models/baseline_best.pkl')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', choices=['fashion_mnist', 'cifar10'])
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    with open(args.model_path, 'rb') as f:
        model_params = pickle.load(f)
    
    if args.dataset == 'fashion_mnist':
        input_size = 784
    else:
        input_size = 3072
    weight_keys = []
    for k in model_params.keys():
        if k.startswith('W'):
            weight_keys.append(k)
    num_layers = len(weight_keys)
    hidden_layers = []
    for i in range(1, num_layers):
        weight_key = 'W' + str(i)
        hidden_size = model_params[weight_key].shape[1]
        hidden_layers.append(hidden_size)
    output_weight_key = 'W' + str(num_layers)
    output_size = model_params[output_weight_key].shape[1]
    
    model = NeuralNetwork(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        activation='relu',
        output_activation='softmax',
        learning_rate=0.001,
        optimizer='adam',
        weight_init='he',
        l2_lambda=0.0001,
        random_seed=42
    )
    model.set_params(model_params)
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if args.dataset == 'fashion_mnist':
        _, _, X_test, y_test = load_fashion_mnist(data_dir)
    else:
        _, _, X_test, y_test = load_cifar10(data_dir)
    
    X_test, y_test = preprocess_data(X_test, y_test, num_classes=output_size, flatten=True, normalize=True)
    evaluate_model(model, X_test, y_test, args.dataset, 'results/plots')


if __name__ == '__main__':
    main()

