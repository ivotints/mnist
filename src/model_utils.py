from keras import layers, models
import tensorflow as tf
import threading
import os

MODEL_PATH = os.path.join('models', 'mnist_model.keras')

class TrainingThread(threading.Thread):
    def __init__(self, model, x_train, y_train_cat, x_test, y_test_cat, epochs, batch_size, history_callback):
        threading.Thread.__init__(self)
        self.model = model
        self.x_train = x_train
        self.y_train_cat = y_train_cat
        self.x_test = x_test
        self.y_test_cat = y_test_cat
        self.epochs = epochs
        self.batch_size = batch_size
        self.history_callback = history_callback
        self.stop_event = threading.Event()

    def run(self):
        history = self.model.fit(
            self.x_train, self.y_train_cat,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.x_test, self.y_test_cat),
            callbacks=[self.history_callback],
            verbose=1
        )
        if not self.stop_event.is_set():
            self.model.save(MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")

    def stop(self):
        self.stop_event.set()

class TrainingHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, update_func):
        super().__init__()
        self.update_func = update_func
        self.accuracy = []
        self.val_accuracy = []
        self.loss = []
        self.val_loss = []
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.accuracy.append(logs.get('accuracy', 0))
        self.val_accuracy.append(logs.get('val_accuracy', 0))
        self.loss.append(logs.get('loss', 0))
        self.val_loss.append(logs.get('val_loss', 0))
        self.update_func(self.accuracy, self.val_accuracy, self.loss, self.val_loss)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def save_model(model):
    model.save(MODEL_PATH)

def load_model():
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH)
