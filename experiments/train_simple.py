import numpy as np
import sys
import os
import pickle
import argparse
import ast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_network import NeuralNetwork
from src.data_loader import load_fashion_mnist, load_cifar10, download_fashion_mnist, download_cifar10, preprocess_data, create_mini_batches, train_val_split
from src.utils import accuracy_score, set_random_seed
from src.hpc_utils import get_data_dir, get_results_dir, setup_hpc_directories
import wandb
from tqdm import tqdm


def labels_to_indices(y):
    if y.ndim > 1 and y.shape[1] > 1:
        return np.argmax(y, axis=1)
    else:
        return y


def load_data(dataset_name, data_dir):
    if dataset_name == 'fashion_mnist':
        load_func = load_fashion_mnist
        download_func = download_fashion_mnist
        input_size = 784
    elif dataset_name == 'cifar10':
        load_func = load_cifar10
        download_func = download_cifar10
        input_size = 3072
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    download_func(data_dir)
    data = load_func(data_dir)
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
        batch_labels = labels_to_indices(y_batch)
        labels.append(batch_labels)
    avg_loss = np.mean(losses)
    all_preds = np.concatenate(predictions)
    all_labels = np.concatenate(labels)
    accuracy = accuracy_score(all_preds, all_labels)
    return avg_loss, accuracy


def evaluate(model, X_val, y_val):
    y_pred_proba = model.predict_proba(X_val)
    y_pred = model.predict(X_val)
    loss = model.compute_loss(y_pred_proba, y_val)
    y_val_indices = labels_to_indices(y_val)
    accuracy = accuracy_score(y_pred, y_val_indices)
    return loss, accuracy


def unwrap_wandb_config(config_dict):
    unwrapped = {}
    for key, value in config_dict.items():
        if isinstance(value, dict) and 'value' in value:
            unwrapped_value = value['value']
            if isinstance(unwrapped_value, dict) and 'value' in unwrapped_value:
                unwrapped_value = unwrapped_value['value']
            unwrapped[key] = unwrapped_value
        else:
            unwrapped[key] = value
    return unwrapped


def get_default_config():
    return {
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
        'dropout_rate': 0.0,
        'val_split': 0.2,
        'random_seed': 42,
        'project_name': 'neural-network-numpy',
        'experiment_name': 'cifar10_baseline',
        'use_wandb': True,
        'entity': 'makssuppras1-danmarks-tekniske-universitet-dtu'
    }


def train(config):
    try:
        run = wandb.run if wandb.run is not None else None
    except:
        run = None
    
    try:
        set_random_seed(config.get('random_seed', 42))
        setup_hpc_directories()
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        data_dir = get_data_dir(project_root)
        X_train_full, y_train_full, X_test, y_test, input_size = load_data(config['dataset'], data_dir)
        
        num_classes = config.get('output_size', 10)
        X_train_full, y_train_full = preprocess_data(X_train_full, y_train_full, num_classes=num_classes, flatten=True, normalize=True)
        X_test, y_test = preprocess_data(X_test, y_test, num_classes=num_classes, flatten=True, normalize=True)
        X_train, X_val, y_train, y_val = train_val_split(X_train_full, y_train_full, val_split=config.get('val_split', 0.2), random_seed=config.get('random_seed', 42))
        
        activation = config.get('activation', 'relu')
        if isinstance(activation, list):
            if len(activation) > 0:
                activation = activation[0]
            else:
                activation = 'relu'
        output_activation = config.get('output_activation', 'softmax')
        if isinstance(output_activation, list):
            if len(output_activation) > 0:
                output_activation = output_activation[0]
            else:
                output_activation = 'softmax'
        
        model = NeuralNetwork(
            input_size=input_size,
            hidden_layers=config['hidden_layers'],
            output_size=num_classes,
            activation=activation,
            output_activation=output_activation,
            learning_rate=config['learning_rate'],
            optimizer=config.get('optimizer', 'adam'),
            weight_init=config.get('weight_init', 'he'),
            l2_lambda=config.get('l2_lambda', 0.0),
            dropout_rate=config.get('dropout_rate', 0.0),
            random_seed=config.get('random_seed', 42)
        )
        
        num_epochs = config.get('num_epochs', 50)
        batch_size = config.get('batch_size', 64)
        best_val_acc = 0.0
        best_model_params = None
        
        for epoch in tqdm(range(num_epochs), desc="Training"):
            train_loss, train_acc = train_epoch(model, X_train, y_train, batch_size)
            val_loss, val_acc = evaluate(model, X_val, y_val)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_params = model.get_params()
            
            if run is not None:
                run.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc})
        
        if best_model_params is not None:
            model.set_params(best_model_params)
            test_loss, test_acc = evaluate(model, X_test, y_test)
            if run is not None:
                run.log({'test_loss': test_loss, 'test_acc': test_acc})
            
            results_dir = get_results_dir(project_root)
            models_dir = os.path.join(results_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            experiment_name = config.get('experiment_name', 'baseline')
            model_filename = experiment_name + '_best.pkl'
            model_path = os.path.join(models_dir, model_filename)
            with open(model_path, 'wb') as f:
                pickle.dump(best_model_params, f)
    except Exception as e:
        if run is not None:
            try:
                run.log({'error': str(e), 'training_failed': True})
                if hasattr(run, 'summary'):
                    run.summary['status'] = 'failed'
                    run.summary['error'] = str(e)
            except:
                pass
        raise
    finally:
        if run is not None:
            try:
                run.finish()
            except:
                pass


def parse_command_line_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--dropout_rate', type=float, default=None)
    parser.add_argument('--hidden_layers', type=str, default=None)
    parser.add_argument('--l2_lambda', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--output_activation', type=str, default=None)
    parser.add_argument('--output_size', type=int, default=None)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--use_wandb', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--val_split', type=float, default=None)
    parser.add_argument('--weight_init', type=str, default=None)
    
    args = parser.parse_args()
    cmd_config = {}
    for key, value in vars(args).items():
        if value is not None:
            if key == 'hidden_layers':
                try:
                    cmd_config[key] = ast.literal_eval(value)
                except:
                    try:
                        value_clean = value.strip('[]')
                        value_parts = value_clean.split(',')
                        hidden_layers_list = []
                        for x in value_parts:
                            hidden_layers_list.append(int(x.strip()))
                        cmd_config[key] = hidden_layers_list
                    except:
                        raise ValueError(f"Could not parse hidden_layers: {value}")
            else:
                cmd_config[key] = value
    return cmd_config


def main():
    default_config = get_default_config()
    cmd_config = parse_command_line_args()
    
    is_sweep_agent = os.environ.get('WANDB_SWEEP_ID') is not None or os.environ.get('WANDB_MODE') == 'sweep' or len(cmd_config) > 0
    
    try:
        run = wandb.init(
            project=default_config.get('project_name', 'neural-network-numpy'),
            config=None if is_sweep_agent else default_config,
            resume='allow'
        )
    except:
        run = None
    
    cfg_override = {}
    try:
        if run is not None:
            wandb_cfg = dict(wandb.config) if hasattr(wandb, 'config') else {}
            cfg_override.update(unwrap_wandb_config(wandb_cfg))
    except:
        pass
    
    if cmd_config:
        cfg_override.update(cmd_config)
    
    config = get_default_config()
    config.update(cfg_override)
    
    required_keys = ['dataset', 'hidden_layers', 'output_size', 'num_epochs', 'batch_size', 'learning_rate']
    missing_keys = []
    for k in required_keys:
        if k not in config:
            missing_keys.append(k)
    if len(missing_keys) > 0:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    if isinstance(config.get('hidden_layers'), dict):
        raise ValueError(f"hidden_layers is still a dict after unwrapping")
    if not isinstance(config.get('hidden_layers'), (list, tuple)):
        raise ValueError(f"hidden_layers must be a list/tuple")
    if isinstance(config.get('learning_rate'), dict):
        raise ValueError(f"learning_rate is still a dict after unwrapping")
    if not isinstance(config.get('learning_rate'), (int, float)):
        raise ValueError(f"learning_rate must be a number")
    
    try:
        train(config)
    except Exception as e:
        if run is not None:
            try:
                if hasattr(run, 'summary'):
                    run.summary['status'] = 'failed'
                    run.summary['error'] = str(e)
                run.finish()
            except:
                pass
        raise

if __name__ == '__main__':
    main()
