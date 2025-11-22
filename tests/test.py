# example_convert.py

from src.quantizedensenet.converter import Converter

# 1. Initialize the converter
converter = Converter()

# --- Define model paths ---
model_name = 'DenseNet121'
base_path = 'models_generated/DenseNet121'
weights_path = 'weights/DenseNet121/raw_weights.h5'
calibration_path = 'data/calibration'

onnx_fp32_path = f'{base_path}/{model_name}_fp32.onnx'
onnx_fp16_path = f'{base_path}/{model_name}_fp16.onnx'
trt_fp32_path = f'{base_path}/{model_name}_fp32.trt'
trt_fp16_path = f'{base_path}/{model_name}_fp16.trt'
trt_int8_path = f'{base_path}/{model_name}_int8.trt'

# --- 2. Convert TF -> ONNX ---

print("Converting TF to ONNX FP32...")
converter.tf_to_onnx(
    input_model=weights_path,
    output_path=onnx_fp32_path,
    precision='fp32',
    only_weigths_of_model=model_name
)

print("Converting TF to ONNX FP16...")
converter.tf_to_onnx(
    input_model=weights_path,
    output_path=onnx_fp16_path,
    precision='fp16',
    only_weigths_of_model=model_name
)

# --- 3. Convert ONNX -> TensorRT ---

# We use the FP32 ONNX model as the base for all TRT builds
base_onnx_model = onnx_fp32_path

print("Building TensorRT FP32 Engine...")
converter.onnx_to_trt(
    input_model=base_onnx_model,
    engine_file_path=trt_fp32_path,
    precision='fp32',
    opt_batch=32,
    max_batch=32
)

print("Building TensorRT FP16 Engine...")
converter.onnx_to_trt(
    input_model=base_onnx_model,
    engine_file_path=trt_fp16_path,
    precision='fp16',
    opt_batch=32,
    max_batch=32
)

print("Building TensorRT INT8 Engine...")
converter.onnx_to_trt(
    input_model=base_onnx_model,
    engine_file_path=trt_int8_path,
    precision='int8',
    calibration_images="data/calibration",  # Path to representative data
    calibration_cache="models_generated/DenseNet121/int8.cache",
    opt_batch=32,
    max_batch=32
)

# --- 4. Convert TF -> TensorRT ---

print("Converting TF to TRT FP32...")
converter.tf_to_trt(
    input_model=weights_path,
    engine_file_path=f'{base_path}/{model_name}_fp32.trt',
    only_weigths_of_model=model_name,
    precision='fp32',
    opt_batch=32,
    max_batch=32
)

print("Converting TF to TRT FP16...")
converter.tf_to_trt(
    input_model=weights_path,
    engine_file_path=f'{base_path}/{model_name}_fp16.trt',
    only_weigths_of_model=model_name,
    precision='fp16',
    opt_batch=32,
    max_batch=32
)

print("\nConverting TF to TRT INT8...")
# (Make sure data/calibration exists first!)
# (Run `python evaluation/prepare_calibration_set.py` first)
converter.tf_to_trt(
    input_model=weights_path,
    engine_file_path=f'{base_path}/{model_name}_int8.trt',
    only_weigths_of_model=model_name,
    precision='int8',
    calibration_images="data/calibration",
    calibration_cache=f"{base_path}/{model_name}_int8.cache",
    opt_batch=32,
    max_batch=32
)

print("All conversions complete.")