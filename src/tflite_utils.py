import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
import time

TFLITE_MODELS_DIR = os.path.join('tflite_models')
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
os.makedirs(TFLITE_MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

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

def save_comparison_results(results, keras_size, float32_size, int8_size):
    """Save comparison results to a text file in assets folder"""
    results_path = os.path.join(ASSETS_DIR, 'tflite_comparison_results.txt')
    
    with open(results_path, 'w') as f:
        f.write("TFLite Model Comparison Results\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Model Accuracy Comparison:\n")
        f.write(f"Original Keras: {results['keras']:.4f}\n")
        f.write(f"TFLite Float32: {results['tflite_float32']:.4f}\n")
        f.write(f"TFLite Int8: {results['tflite_int8']:.4f}\n\n")
        
        # Calculate accuracy drops
        float32_drop = results['keras'] - results['tflite_float32']
        int8_drop = results['keras'] - results['tflite_int8']
        
        f.write(f"Accuracy drop Float32: {float32_drop:.4f}\n")
        f.write(f"Accuracy drop Int8: {int8_drop:.4f}\n\n")
        
        f.write("Model Sizes:\n")
        f.write(f"Keras: {keras_size:.2f} MB\n")
        f.write(f"TFLite Float32: {float32_size:.2f} MB\n")
        f.write(f"TFLite Int8: {int8_size:.2f} MB\n\n")
        
        f.write("Compression ratios:\n")
        f.write(f"Float32/Keras: {float32_size/keras_size:.2f}x\n")
        f.write(f"Int8/Keras: {int8_size/keras_size:.2f}x\n")
        f.write(f"Int8/Float32: {int8_size/float32_size:.2f}x\n")
    
    print(f"Comparison results saved to {results_path}")
    return results_path

def benchmark_model(interpreter, test_data, num_runs=100):
    """Benchmark inference time for a TFLite model"""
    input_details = interpreter.get_input_details()
    input_dtype = input_details[0]['dtype']
    
    # Prepare test data based on model input requirements
    processed_test_data = []
    for sample in test_data[:num_runs]:
        if input_dtype == np.int8:
            # For int8 quantized model, scale the input
            input_scale, input_zero_point = input_details[0]["quantization"]
            processed_sample = sample / input_scale + input_zero_point
            processed_sample = np.clip(processed_sample, -128, 127).astype(np.int8)
        else:
            # For float32 models, use as is
            processed_sample = sample.astype(np.float32)
        processed_test_data.append(processed_sample)
    
    processed_test_data = np.array(processed_test_data)
    
    # Warm up
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], processed_test_data[0:1])
        interpreter.invoke()
    
    # Benchmark
    times = []
    for i in range(len(processed_test_data)):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], processed_test_data[i:i+1])
        interpreter.invoke()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'times': times
    }

def benchmark_keras_model(model, test_data, num_runs=100):
    """Benchmark inference time for a Keras model"""
    # Warm up
    for _ in range(10):
        model.predict(test_data[0:1], verbose=0)
    
    # Benchmark
    times = []
    for i in range(min(num_runs, len(test_data))):
        start_time = time.time()
        model.predict(test_data[i:i+1], verbose=0)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'times': times
    }

def run_performance_tests(keras_model, tflite_float32_path, tflite_int8_path, test_data):
    """Run performance tests for all model formats"""
    results = {}
    
    print("Running performance tests...")
    
    # Test Keras model
    print("Testing Keras model...")
    keras_perf = benchmark_keras_model(keras_model, test_data)
    results['keras'] = keras_perf
    
    # Test TFLite Float32
    print("Testing TFLite Float32...")
    interpreter_float32 = load_tflite_model(tflite_float32_path)
    float32_perf = benchmark_model(interpreter_float32, test_data)
    results['tflite_float32'] = float32_perf
    
    # Test TFLite Int8
    print("Testing TFLite Int8...")
    interpreter_int8 = load_tflite_model(tflite_int8_path)
    int8_perf = benchmark_model(interpreter_int8, test_data)
    results['tflite_int8'] = int8_perf
    
    return results

def save_performance_results(perf_results):
    """Save performance test results to a text file"""
    results_path = os.path.join(ASSETS_DIR, 'performance_test_results.txt')
    
    with open(results_path, 'w') as f:
        f.write("TFLite Performance Test Results\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Inference Time (milliseconds per prediction)\n")
        f.write("-" * 50 + "\n")
        
        for model_name, results in perf_results.items():
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"  Mean: {results['mean_time']:.3f} ms\n")
            f.write(f"  Std:  {results['std_time']:.3f} ms\n")
            f.write(f"  Min:  {results['min_time']:.3f} ms\n")
            f.write(f"  Max:  {results['max_time']:.3f} ms\n")
        
        # Calculate speedup ratios
        keras_time = perf_results['keras']['mean_time']
        float32_time = perf_results['tflite_float32']['mean_time']
        int8_time = perf_results['tflite_int8']['mean_time']
        
        f.write("\nSpeedup Ratios (lower is better):\n")
        f.write(f"  Float32/Keras: {keras_time/float32_time:.2f}x\n")
        f.write(f"  Int8/Keras: {keras_time/int8_time:.2f}x\n")
        f.write(f"  Int8/Float32: {float32_time/int8_time:.2f}x\n")
    
    print(f"Performance results saved to {results_path}")
    return results_path

def load_existing_tflite_model(model_path):
    """Load an existing TFLite model from file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter