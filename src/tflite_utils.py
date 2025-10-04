import tensorflow as tf
import os
import numpy as np
from tensorflow import keras

TFLITE_MODELS_DIR = os.path.join('tflite_models')
os.makedirs(TFLITE_MODELS_DIR, exist_ok=True)

def convert_to_tflite_float32(model, model_name='mnist_model_float32.tflite'):
    """Convert Keras model to TFLite with float32 precision"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = os.path.join(TFLITE_MODELS_DIR, model_name)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_path, len(tflite_model)

def convert_to_tflite_int8(model, x_train, model_name='mnist_model_int8.tflite'):
    """Convert Keras model to TFLite with int8 quantization"""
    def representative_dataset():
        # Use a subset of training data for calibration
        for data in x_train[:1000]:  # Use first 1000 samples for calibration
            yield [np.expand_dims(data, axis=0).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = os.path.join(TFLITE_MODELS_DIR, model_name)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_path, len(tflite_model)

def load_tflite_model(model_path):
    """Load TFLite model"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_with_tflite(interpreter, input_data):
    """Make prediction with TFLite model"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess input based on model requirements
    if input_details[0]['dtype'] == np.int8:
        # For int8 quantized model, scale the input
        input_scale, input_zero_point = input_details[0]["quantization"]
        input_data = input_data / input_scale + input_zero_point
        input_data = np.clip(input_data, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Postprocess output for int8 model
    if output_details[0]['dtype'] == np.int8:
        output_scale, output_zero_point = output_details[0]["quantization"]
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    return output_data

def get_model_size_mb(model_path):
    """Get model size in MB"""
    return os.path.getsize(model_path) / (1024 * 1024)

def compare_models(original_model, tflite_float32_path, tflite_int8_path, x_test, y_test):
    """Compare accuracy of different model formats"""
    results = {}

    # Original Keras model accuracy
    keras_predictions = original_model.predict(x_test)
    keras_accuracy = np.mean(np.argmax(keras_predictions, axis=1) == y_test)
    results['keras'] = keras_accuracy

    # TFLite float32 accuracy
    interpreter_float32 = load_tflite_model(tflite_float32_path)
    tflite_float32_predictions = []
    for sample in x_test:
        pred = predict_with_tflite(interpreter_float32, np.expand_dims(sample, axis=0))
        tflite_float32_predictions.append(pred[0])
    tflite_float32_predictions = np.array(tflite_float32_predictions)
    tflite_float32_accuracy = np.mean(np.argmax(tflite_float32_predictions, axis=1) == y_test)
    results['tflite_float32'] = tflite_float32_accuracy

    # TFLite int8 accuracy
    interpreter_int8 = load_tflite_model(tflite_int8_path)
    tflite_int8_predictions = []
    for sample in x_test:
        pred = predict_with_tflite(interpreter_int8, np.expand_dims(sample, axis=0))
        tflite_int8_predictions.append(pred[0])
    tflite_int8_predictions = np.array(tflite_int8_predictions)
    tflite_int8_accuracy = np.mean(np.argmax(tflite_int8_predictions, axis=1) == y_test)
    results['tflite_int8'] = tflite_int8_accuracy

    return results