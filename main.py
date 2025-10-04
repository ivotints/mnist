import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os

from src.data_utils import load_mnist_data, visualize_mnist_examples, visualize_class_distribution
from src.model_utils import create_model, TrainingThread, TrainingHistoryCallback, load_model, save_model, MODEL_PATH
from src.drawing_canvas import DrawingCanvas
from src.tflite_utils import convert_to_tflite_float32, convert_to_tflite_int8, compare_models, get_model_size_mb

os.makedirs('models', exist_ok=True)

if __name__ == "__main__":
    (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat) = load_mnist_data()

    root = tk.Tk()
    root.title("MNIST Visualization and Training")
    root.geometry("800x600")
    root.protocol("WM_DELETE_WINDOW", root.quit)

    notebook = ttk.Notebook(root)
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    tab3 = ttk.Frame(notebook)
    tab4 = ttk.Frame(notebook)
    tab5 = ttk.Frame(notebook)
    notebook.add(tab1, text="MNIST Examples")
    notebook.add(tab2, text="Class Distribution")
    notebook.add(tab3, text="Model Training")
    notebook.add(tab4, text="Draw & Predict")
    notebook.add(tab5, text="TFLite Export")
    notebook.pack(expand=1, fill="both")

    # Add MNIST examples to the first tab
    fig1 = visualize_mnist_examples(x_train, y_train)
    canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Add class distribution visualization to the second tab
    fig2 = visualize_class_distribution(y_train, y_test)
    canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Training tab components
    training_frame = ttk.Frame(tab3)
    training_frame.pack(fill=tk.BOTH, expand=1)
    control_frame = ttk.Frame(training_frame)
    control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    ttk.Label(control_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=5)
    epoch_var = tk.StringVar(value="5")
    epoch_entry = ttk.Entry(control_frame, width=5, textvariable=epoch_var)
    epoch_entry.grid(row=0, column=1, padx=5, pady=5)
    ttk.Label(control_frame, text="Batch Size:").grid(row=0, column=2, padx=5, pady=5)
    batch_var = tk.StringVar(value="32")
    batch_entry = ttk.Entry(control_frame, width=5, textvariable=batch_var)
    batch_entry.grid(row=0, column=3, padx=5, pady=5)
    fig_training = plt.figure(figsize=(10, 4))
    ax1 = fig_training.add_subplot(121)
    ax2 = fig_training.add_subplot(122)
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    fig_training.tight_layout()
    canvas_training = FigureCanvasTkAgg(fig_training, master=training_frame)
    canvas_training.draw()
    canvas_training.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    training_thread = None
    model = None
    def update_training_plots(acc, val_acc, loss, val_loss):
        ax1.clear()
        ax2.clear()
        ax1.plot(acc, label='Training')
        ax1.plot(val_acc, label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax2.plot(loss, label='Training')
        ax2.plot(val_loss, label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        fig_training.tight_layout()
        canvas_training.draw()
    def start_training():
        global training_thread, model
        try:
            epochs = int(epoch_var.get())
            batch_size = int(batch_var.get())
        except ValueError:
            messagebox.showerror("Error", "Epochs and Batch Size must be integers")
            return
        start_button.configure(state="disabled")
        stop_button.configure(state="normal")
        model = create_model()
        history_callback = TrainingHistoryCallback(update_training_plots)
        training_thread = TrainingThread(
            model, x_train, y_train_cat, x_test, y_test_cat,
            epochs, batch_size, history_callback
        )
        training_thread.start()
        check_thread_status()
    def stop_training():
        global training_thread
        if training_thread and training_thread.is_alive():
            training_thread.stop()
            training_thread.join()
            messagebox.showinfo("Training Stopped", "Training has been stopped")
        start_button.configure(state="normal")
        stop_button.configure(state="disabled")
    def check_thread_status():
        global training_thread
        if training_thread and training_thread.is_alive():
            root.after(100, check_thread_status)
        else:
            start_button.configure(state="normal")
            stop_button.configure(state="disabled")
    def predict_digit(img_array, canvas_obj):
        global model
        if model is None:
            try:
                model = load_model()
                print(f"Model loaded from {MODEL_PATH}")
            except Exception as e:
                messagebox.showerror("Error", f"Please train a model first. {e}")
                return
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        canvas_obj.update_prediction(predicted_class, predictions[0])
    start_button = ttk.Button(control_frame, text="Start Training", command=start_training)
    start_button.grid(row=0, column=4, padx=5, pady=5)
    stop_button = ttk.Button(control_frame, text="Stop Training", command=stop_training, state="disabled")
    stop_button.grid(row=0, column=5, padx=5, pady=5)
    drawing_canvas = DrawingCanvas(tab4, predict_digit)

    # TFLite Export tab
    tflite_frame = ttk.Frame(tab5)
    tflite_frame.pack(fill=tk.BOTH, expand=1)

    # Export controls
    export_frame = ttk.LabelFrame(tflite_frame, text="Export Models", padding=10)
    export_frame.pack(fill=tk.X, padx=10, pady=10)

    ttk.Button(export_frame, text="Export Float32 TFLite", command=lambda: export_float32()).grid(row=0, column=0, padx=5, pady=5)
    ttk.Button(export_frame, text="Export Int8 TFLite", command=lambda: export_int8()).grid(row=0, column=1, padx=5, pady=5)
    ttk.Button(export_frame, text="Compare Models", command=lambda: compare_model_performance()).grid(row=0, column=2, padx=5, pady=5)

    # Results display
    results_text = tk.Text(tflite_frame, height=15, wrap=tk.WORD)
    results_text.pack(fill=tk.BOTH, expand=1, padx=10, pady=10)

    # Global variables for TFLite models
    tflite_float32_path = None
    tflite_int8_path = None

    def export_float32():
        global model, tflite_float32_path
        if model is None:
            try:
                model = load_model()
            except:
                messagebox.showerror("Error", "Please train a model first")
                return

        try:
            tflite_float32_path, size_bytes = convert_to_tflite_float32(model)
            size_mb = size_bytes / (1024 * 1024)
            results_text.insert(tk.END, f"Float32 TFLite model exported: {tflite_float32_path}\n")
            results_text.insert(tk.END, f"Size: {size_mb:.2f} MB\n\n")
            results_text.see(tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export Float32 model: {e}")

    def export_int8():
        global model, tflite_int8_path
        if model is None:
            try:
                model = load_model()
            except:
                messagebox.showerror("Error", "Please train a model first")
                return

        try:
            tflite_int8_path, size_bytes = convert_to_tflite_int8(model, x_train)
            size_mb = size_bytes / (1024 * 1024)
            results_text.insert(tk.END, f"Int8 TFLite model exported: {tflite_int8_path}\n")
            results_text.insert(tk.END, f"Size: {size_mb:.2f} MB\n\n")
            results_text.see(tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export Int8 model: {e}")

    def compare_model_performance():
        global model, tflite_float32_path, tflite_int8_path
        if model is None:
            messagebox.showerror("Error", "Please train a model first")
            return
        if tflite_float32_path is None or tflite_int8_path is None:
            messagebox.showerror("Error", "Please export both Float32 and Int8 models first")
            return

        try:
            results_text.insert(tk.END, "Comparing model performance...\n")
            results = compare_models(model, tflite_float32_path, tflite_int8_path, x_test, y_test)

            results_text.insert(tk.END, "\nModel Accuracy Comparison:\n")
            results_text.insert(tk.END, f"Original Keras: {results['keras']:.4f}\n")
            results_text.insert(tk.END, f"TFLite Float32: {results['tflite_float32']:.4f}\n")
            results_text.insert(tk.END, f"TFLite Int8: {results['tflite_int8']:.4f}\n")

            # Calculate accuracy drops
            float32_drop = results['keras'] - results['tflite_float32']
            int8_drop = results['keras'] - results['tflite_int8']

            results_text.insert(tk.END, f"\nAccuracy drop Float32: {float32_drop:.4f}\n")
            results_text.insert(tk.END, f"Accuracy drop Int8: {int8_drop:.4f}\n")

            # Model sizes
            keras_size = get_model_size_mb(MODEL_PATH)
            float32_size = get_model_size_mb(tflite_float32_path)
            int8_size = get_model_size_mb(tflite_int8_path)

            results_text.insert(tk.END, f"\nModel Sizes:\n")
            results_text.insert(tk.END, f"Keras: {keras_size:.2f} MB\n")
            results_text.insert(tk.END, f"TFLite Float32: {float32_size:.2f} MB\n")
            results_text.insert(tk.END, f"TFLite Int8: {int8_size:.2f} MB\n")

            results_text.insert(tk.END, f"\nCompression ratios:\n")
            results_text.insert(tk.END, f"Float32/Keras: {float32_size/keras_size:.2f}x\n")
            results_text.insert(tk.END, f"Int8/Keras: {int8_size/keras_size:.2f}x\n")
            results_text.insert(tk.END, f"Int8/Float32: {int8_size/float32_size:.2f}x\n\n")

            results_text.see(tk.END)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to compare models: {e}")

    root.mainloop()