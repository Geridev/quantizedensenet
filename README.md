# quantizedensenet

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://tensorflow.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.14%2B-76B900.svg)](https://developer.nvidia.com/tensorrt)

A Python package for seamlessly converting DenseNet TensorFlow models to ONNX and TensorRT formats, enabling optimized inference on NVIDIA GPUs.

## Features

- **TensorFlow to ONNX Conversion**: Convert SavedModel or model weights to ONNX format with FP32 or FP16 precision.
- **ONNX to TensorRT Conversion**: Build optimized TensorRT engines with dynamic batch sizes.
- **Direct TF to TRT Pipeline**: One-step conversion from TensorFlow to TensorRT.
- **INT8 Quantization Support**: Improve inference speed with INT8 calibration using image directories or cache files.
- **DenseNet Support**: specialized support for DenseNet121 and DenseNet201 architectures.
- **Comprehensive Logging**: Detailed conversion process tracking and validation.

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Install from Wheel File](#install-from-wheel-file)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Converter](#converter-class)
  - [Keep In Mind](#keep-in-mind)
  - [TensorFlow to ONNX](#tf_to_onnx)
  - [ONNX to TensorRT](#onnx_to_trt)
  - [TensorFlow to TensorRT](#tf_to_trt)

## Installation

### Prerequisites

Before installing the package, ensure you have the following:

- **Python 3.8 or higher** - Check with `python --version`
- **pip** (Python package manager) - Check with `pip --version`
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit 12.x** - (Required by `tensorrt_cu12` and `cuda_bindings==12.9.2`)
- **TensorRT 10.14+**

### Install from Wheel File

1. **Download the wheel file** provided to you.
2. **Open a terminal** and navigate to the directory containing the `.whl` file.
3. **Install the package** using pip.
4. **If dependencies are not automatically installed** with the wheel file, you can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Quick Start
1. Install the wheel and dependencies, then import the converter: `from quantizedensenet import Converter`

2. Create a converter instance: `converter = Converter()`

3. Convert TensorFlow to ONNX or straight to TensorRT, with optional FP16/INT8 settings and dynamic batching.


## Usage Examples

### Converter Class
The `Converter` class provides three utilities: convert TensorFlow models to ONNX, build TensorRT engines from ONNX, and run a one-call TensorFlow to TensorRT pipeline for deployment-ready inference engines.

### Keep In Mind

* The original input shape $(N, H, W, C)$ will be changed to $(N, C, H, W)$ for optimized execution.

* All the functions' output_path or engine_file_path could be None; this way the functions return the created models/engines in memory.

* **Memory Management**: It is not recommended to use the converted models for inference immediately after conversion in the same script. The best practice is to restart the Python kernel to free up allocated CUDA memory. After that, with a new run, you can load the engine and run inference without errors.

* FP16 TensorRT engines are generally the best choice, as they provide the fastest inference without significant accuracy loss.

### tf_to_onnx

* Converts a `tf.keras` SavedModel or a `.h5` Weights file to an ONNX model.

* If you pass a `.h5` file that only contains weights, you must specify the `only_weigths_of_model` argument (Supports: `'DenseNet121'`, `'DenseNet201'`).

* The keras model input shape must be `(None, 224, 224, 3)`.

* Supports exporting to FP32 or FP16 ONNX graph.

```python
from quantizedensenet import Converter

converter = Converter()
onnx_model = converter.tf_to_onnx(
    input_model="path/to/densenet121_weights.h5",
    output_path="path/to/output/model.onnx",
    precision="fp16",                  
    only_weigths_of_model='DenseNet121', # Specify base model if using weights only
    opset=13
)
```

### onnx_to_trt

* Parses an ONNX model and builds a TensorRT engine.

* The input model can be passed as a path to a `.onnx` file or as a `onnx.ModelProto` object.

* The ONNX model input shape must be `(-1, 3, 224, 224)`.

* If engine_file_path is None and `auto_generate_engine_path=True`, it auto-generates the path based on the input filename.

* Supports exporting to **FP32**, **FP16**, or **INT8** TRT engines.

* **Dynamic Batching**: You must specify 3 arguments:

  * `min_batch`: The minimum number of images a batch could ever contain (usually 1).

  * `opt_batch`: The most common batch size for inference (should be close to max_batch).

  * `max_batch`: The maximum number of images a batch could ever contain.

```python
from quantizedensenet import Converter

converter = Converter()
engine = converter.onnx_to_trt(
    input_model="path/to/model.onnx",
    engine_file_path="path/to/output/model.trt",
    precision="fp16",                    
    min_batch=1,
    opt_batch=16,
    max_batch=32
)
```

* **INT8 Calibration**: INT8 mode requires calibration data or an existing cache.

* You can provide:

  * A directory path containing images.

  * A single image path.

  * A list of image paths.

  * A path to a calibration cache file.

* If `calibration_cache` is provided but does not exist, it will be created using the `calibration_images`.

```python
from quantizedensenet import Converter

converter = Converter()
model = converter.onnx_to_trt(
    input_model="path/to/model.onnx",
    engine_file_path="path/to/int8_densenet.trt",
    min_batch=1,
    max_batch=32,
    opt_batch=32,
    precision="int8",
    calibration_images="path/to/calibration/images/dir",
    calibration_cache="path/to/calibration.cache",
)
```

### tf_to_trt

Runs the end-to-end pipeline in one call. Exports a TensorFlow model to ONNX, then builds a TensorRT engine with the selected precision and batch profiles, streamlining deployment.

* If you pass a `.h5` file that only contains weights, you should also specify the base model using `only_weigths_of_model`.

* When using INT8, provide `calibration_images` or a `calibration_cache`.

```python
from quantizedensenet import Converter

converter = Converter()
model = converter.tf_to_trt(
    input_model="path/to/densenet201_weights.h5",
    engine_file_path="path/to/int8_densenet201.trt",
    only_weigths_of_model='DenseNet201',
    min_batch=1,
    opt_batch=32,
    max_batch=32,
    precision="int8",
    calibration_images="path/to/calibration/images/dir",
    calibration_cache="path/to/calibration.cache",
    auto_generate_engine_path=False
)
```

