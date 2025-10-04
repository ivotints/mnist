import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os

# Create assets directory
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat)

def visualize_mnist_examples(x_train, y_train):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        idx = np.where(y_train == i)[0][0]
        plt.imshow(x_train[idx].reshape(28, 28), cmap='gray')
        plt.title(f"Digit: {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'mnist_examples.png'))
    return plt.gcf()

def visualize_class_distribution(y_train, y_test):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    plt.bar(unique_train, counts_train)
    plt.title('Distribution of classes in train sample')
    plt.xlabel('Digit')
    plt.ylabel('Amount')
    plt.subplot(1, 2, 2)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    plt.bar(unique_test, counts_test)
    plt.title('Distribution of classes in test sample')
    plt.xlabel('Digit')
    plt.ylabel('Amount')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'class_distribution.png'))
    return plt.gcf()
