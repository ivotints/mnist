import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw

class DrawingCanvas:
    def __init__(self, master, predict_callback):
        self.master = master
        self.predict_callback = predict_callback
        self.frame = ttk.Frame(master)
        self.frame.pack(fill=tk.BOTH, expand=1)
        self.canvas = tk.Canvas(self.frame, width=280, height=280, bg='black')
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.button_frame = ttk.Frame(self.frame)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        self.clear_button = ttk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(pady=5)
        self.prediction_label = ttk.Label(self.button_frame, text="Draw a digit")
        self.prediction_label.pack(pady=5)
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.button_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(pady=5)
        self.previous_x = None
        self.previous_y = None
        self.image = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image)
    def paint(self, event):
        x, y = event.x, event.y
        if self.previous_x and self.previous_y:
            self.canvas.create_line(self.previous_x, self.previous_y, x, y, fill="white", width=20, capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.previous_x, self.previous_y, x, y], fill="white", width=20, joint="curve")
        self.previous_x = x
        self.previous_y = y
    def on_release(self, event):
        self.previous_x = None
        self.previous_y = None
        self.predict()
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.configure(text="Draw a digit")
        self.ax.clear()
        self.canvas_widget.draw()
    def predict(self):
        img = self.image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        if self.predict_callback:
            self.predict_callback(img_array, self)
    def update_prediction(self, prediction, probabilities):
        self.prediction_label.configure(text=f"Prediction: {prediction}")
        self.ax.clear()
        self.ax.bar(range(10), probabilities)
        self.ax.set_xlabel('Digit')
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Prediction Probabilities')
        self.ax.set_xticks(range(10))
        self.canvas_widget.draw()
