import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import (
    download_fashion_mnist,
    download_cifar10,
    load_fashion_mnist,
    load_cifar10,
    preprocess_data,
    create_mini_batches,
    train_val_split
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def test_fashion_mnist():
    download_fashion_mnist(DATA_DIR)
    data = load_fashion_mnist(DATA_DIR)
    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]
    
    assert X_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000,)
    assert X_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000,)
    
    assert X_train.min() >= 0
    assert X_train.max() <= 255
    assert y_train.min() >= 0
    assert y_train.max() <= 9
    
    X_proc, y_proc = preprocess_data(X_train[:100], y_train[:100], flatten=True, normalize=True)
    
    assert X_proc.shape == (100, 784)
    assert y_proc.shape == (100, 10)
    assert X_proc.min() >= 0
    assert X_proc.max() <= 1
    
    batches = create_mini_batches(X_proc, y_proc, batch_size=32)
    assert len(batches) == 4
    return True

def test_cifar10():
    download_cifar10(DATA_DIR)
    data = load_cifar10(DATA_DIR)
    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]
    
    assert X_train.shape == (50000, 32, 32, 3)
    assert y_train.shape == (50000,)
    assert X_test.shape == (10000, 32, 32, 3)
    assert y_test.shape == (10000,)
    
    assert X_train.min() >= 0
    assert X_train.max() <= 255
    assert y_train.min() >= 0
    assert y_train.max() <= 9
    
    X_proc, y_proc = preprocess_data(X_train[:100], y_train[:100], flatten=True, normalize=True)
    
    assert X_proc.shape == (100, 3072)
    assert y_proc.shape == (100, 10)
    assert X_proc.min() >= 0
    assert X_proc.max() <= 1
    
    X_tr, X_val, y_tr, y_val = train_val_split(X_proc, y_proc, val_split=0.2, random_seed=42)
    
    assert X_tr.shape[0] == 80
    assert X_val.shape[0] == 20
    
    return True

def main():
    test_fashion_mnist()
    test_cifar10()

if __name__ == "__main__":
    main()

