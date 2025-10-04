import numpy as np
from tensorflow import keras
from keras import layers, models
import os
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

os.makedirs('models', exist_ok=True)

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)

def visualize_mnist_examples(x_train, y_train):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)

        idx = np.where(y_train == i)[0][0]
        plt.imshow(x_train[idx], cmap='gray')
        plt.title(f"Digit: {i}")
        plt.axis('off')
    # plt.tight_layout()
    plt.savefig('mnist_examples.png')
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
    unique_train, counts_train = np.unique(y_test, return_counts=True)
    plt.bar(unique_train, counts_train)
    plt.title('Distribution of classes in test sample')
    plt.xlabel('Digit')
    plt.ylabel('Amount')

    # plt.tight_layout()
    plt.savefig('mnist_examples.png')
    return plt.gcf()



if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    root = tk.Tk()
    root.title("MNIST Visualisation")
    root.geometry("800x600")

    notebook = ttk.Notebook(root)
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Visualization of data")
    notebook.pack(expand=1, fill="both")

    fig1 = visualize_mnist_examples(x_train, y_train)
    canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    fig2 = visualize_class_distribution(y_train, y_test)
    canvas2 = FigureCanvasTkAgg(fig2, master=tab1)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    root.mainloop()