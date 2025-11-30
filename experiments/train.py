# Main training script for neural network experiments with WandB logging

import numpy as np
import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.data_loader import load_fashion_mnist, load_cifar10, preprocess_data, create_mini_batches, train_val_split
from src.utils import accuracy_score, set_random_seed
from src.hpc_utils import get_data_dir, get_results_dir, setup_hpc_directories
import wandb
from tqdm import tqdm

def load_data(dataset_name, data_dir):
    if dataset_name == 'fashion_mnist':
        data = load_fashion_mnist(data_dir)
        input_size = 784
    elif dataset_name == 'cifar10':
        data = load_cifar10(data_dir)
        input_size = 3072
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X_train_full = data[0]
    y_train_full = data[1]
    X_test = data[2]
    y_test = data[3]
    return X_train_full, y_train_full, X_test, y_test, input_size


def train_epoch(model, X_train, y_train, batch_size):
    batches = create_mini_batches(X_train, y_train, batch_size=batch_size, shuffle=True)
    losses = []
    predictions = []
    labels = []
    
    for X_batch, y_batch in batches:
        loss = model.train_step(X_batch, y_batch)
        losses.append(loss)
        
        preds = model.predict(X_batch)
        predictions.append(preds)
        
        if y_batch.ndim > 1:
            y_batch_indices = np.argmax(y_batch, axis=1)
        else:
            y_batch_indices = y_batch
        labels.append(y_batch_indices)
    
    avg_loss = np.mean(losses)
    all_preds = np.concatenate(predictions)
    all_labels = np.concatenate(labels)
    accuracy = accuracy_score(all_preds, all_labels)
    
    return avg_loss, accuracy


def evaluate(model, X_val, y_val):
    y_pred_proba = model.predict_proba(X_val)
    y_pred = model.predict(X_val)
    
    loss = model.compute_loss(y_pred_proba, y_val)
    
    if y_val.ndim > 1 and y_val.shape[1] > 1:
        y_val_indices = np.argmax(y_val, axis=1)
    else:
        y_val_indices = y_val
    
    accuracy = accuracy_score(y_pred, y_val_indices)
    return loss, accuracy


def train(config=None):
    use_wandb = False
    if config is None:
        wandb.init()
        config = dict(wandb.config)
        use_wandb = True
    else:
        use_wandb = config.get('use_wandb', True)
        if use_wandb:
            try:
                wandb.init(
                    project=config.get('project_name', 'neural-network-numpy'),
                    name=config.get('experiment_name', 'baseline'),
                    entity=config.get('entity', None),
                    config=config
                )
            except Exception as e:
                print(f"Warning: WandB init failed ({e}). Continuing without WandB.")
                use_wandb = False
    
    set_random_seed(config.get('random_seed', 42))
    setup_hpc_directories()
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = get_data_dir(project_root)
    X_train_full, y_train_full, X_test, y_test, input_size = load_data(config['dataset'], data_dir)
    
    num_classes = config.get('output_size', 10)
    X_train_full, y_train_full = preprocess_data(X_train_full, y_train_full, num_classes=num_classes, flatten=True, normalize=True)
    X_test, y_test = preprocess_data(X_test, y_test, num_classes=num_classes, flatten=True, normalize=True)
    
    val_split = config.get('val_split', 0.2)
    random_seed = config.get('random_seed', 42)
    X_train, X_val, y_train, y_val = train_val_split(X_train_full, y_train_full, val_split=val_split, random_seed=random_seed)
    
    model = NeuralNetwork(
        input_size=input_size,
        hidden_layers=config['hidden_layers'],
        output_size=num_classes,
        activation=config.get('activation', 'relu'),
        output_activation=config.get('output_activation', 'softmax'),
        learning_rate=config['learning_rate'],
        optimizer=config.get('optimizer', 'adam'),
        weight_init=config.get('weight_init', 'he'),
        l2_lambda=config.get('l2_lambda', 0.0),
        dropout_rate=config.get('dropout_rate', 0.0),
        random_seed=random_seed
    )
    
    best_val_acc = 0.0
    best_model_params = None
    num_epochs = config.get('num_epochs', 50)
    batch_size = config.get('batch_size', 64)
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss, train_acc = train_epoch(model, X_train, y_train, batch_size)
        val_loss, val_acc = evaluate(model, X_val, y_val)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_params = model.get_params()
        
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })
    
    if best_model_params is not None:
        model.set_params(best_model_params)
    
    test_loss, test_acc = evaluate(model, X_test, y_test)
    if use_wandb:
        wandb.log({'test_loss': test_loss, 'test_acc': test_acc})
    
    results_dir = get_results_dir(project_root)
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    experiment_name = config.get('experiment_name', 'baseline')
    model_filename = experiment_name + '_best.pkl'
    model_path = os.path.join(models_dir, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_params, f)
    
    if use_wandb:
        wandb.finish()


def main():
    config = {
        'dataset': 'cifar10',
        'hidden_layers': [1024, 512, 256],
        'output_size': 10,
        'activation': 'relu',
        'output_activation': 'softmax',
        'num_epochs': 150,
        'batch_size': 64,
        'learning_rate': 0.0003,
        'optimizer': 'adam',
        'l2_lambda': 0.0001,
        'weight_init': 'he',
        'val_split': 0.2,
        'random_seed': 42,
        'project_name': 'neural-network-numpy',
        'experiment_name': 'cifar10_baseline',
        'use_wandb': True,
        'entity': 'makssuppras1-danmarks-tekniske-universitet-dtu'
    }
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=config['dataset'])
    parser.add_argument('--epochs', type=int, default=config['num_epochs'])
    parser.add_argument('--batch-size', type=int, default=config['batch_size'])
    parser.add_argument('--lr', type=float, default=config['learning_rate'])
    parser.add_argument('--optimizer', type=str, default=config['optimizer'])
    parser.add_argument('--name', type=str, default=config['experiment_name'])
    parser.add_argument('--hidden-layers', type=str, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    
    args = parser.parse_args()
    
    config['dataset'] = args.dataset
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['optimizer'] = args.optimizer
    config['experiment_name'] = args.name
    
    if args.hidden_layers is not None:
        hidden_layers_str = args.hidden_layers
        hidden_layers_list = hidden_layers_str.split(',')
        hidden_layers = []
        for x in hidden_layers_list:
            hidden_layers.append(int(x.strip()))
        config['hidden_layers'] = hidden_layers
    
    if args.no_wandb:
        config['use_wandb'] = False
    
    train(config)


if __name__ == '__main__':
    main()
