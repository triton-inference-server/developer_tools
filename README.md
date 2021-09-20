<!--
Copyright (c) 2021, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# The RAPIDS-Triton Library

This project is designed to make it easy to integrate any C++-based algorithm
into the NVIDIA Triton Inference Server. Originally developed to assist with
the integration of RAPIDS algorithms, this library can be used by anyone to
quickly get up and running with a custom backend for Triton.

## Background

### Triton

The [NVIDIA Triton Inference
Server](https://developer.nvidia.com/nvidia-triton-inference-server) offers a
complete open-source solution for deployment of machine learning models from a
wide variety of ML frameworks (PyTorch, Tensorflow, ONNX, XGBoost, etc.) on
both CPU and GPU hardware. It allows you to maximize inference performance in
production (whether that means maximizing throughput, minimizing latency, or
optimizing some other metric) regardless of how you may have trained your ML
model. Through smart batching, efficient pipeline handling, and tools to
simplify deployments almost anywhere, Triton helps make production inference
serving simpler and more cost-effective.

### Custom Backends

While Triton natively supports many common ML frameworks, you may wish to take
advantage of Triton's features for something a little more specialized. Triton
provides support for different kinds of models via "backends:" modular
libraries which provide the specialized logic for those models. Triton allows
you to create custom backends in 
[Python](https://github.com/triton-inference-server/python_backend), but for
those who wish to use C++ directly, RAPIDS-Triton can help simplify the process
of developing your backend.

The goal of RAPIDS-Triton is not to facilitate every possible use case of the
Triton backend API but to make the most common uses of this API easier by
providing a simpler interface to them. That being said, if there is a feature
of the Triton backend API which RAPIDS-Triton does not expose and which you
wish to use in a custom backend, please [submit a feature
request](https://github.com/rapidsai/rapids-triton/issues), and we will see if
it can be added.

## Simple Example

In the `cpp/src` directory of this repository, you can see a complete,
annotated example of a backend built with RAPIDS-Triton. The core of any
backend is defining the `predict` function for your model as shown below:

```
  void predict(rapids::Batch& batch) const {
    rapids::Tensor<float> input = get_input<float>(batch, "input__0");
    rapids::Tensor<float> output = get_output<float>(batch, "output__0");

    rapids::copy<float const>(output, input);

    output.finalize();
  }
```

In this example, we ask Triton to provide a tensor named `"input__0"` and copy
it to an output tensor named `"output__0"`. Thus, our "inference" function in
this simple example is just a passthrough from one input tensor to one output
tensor.

To do something more sophisticated in this `predict` function, we might take
advantage of the `data()` method of Tensor objects, which provides a raw
pointer (on host or device) to the underlying data along with `size()`, and
`mem_type()` to determine the number of elements in the Tensor and whether they
are stored on host or device respectively. Note that `finalize()` must be
called on all output tensors before returning from the predict function.

For a much more detailed look at developing backends with RAPIDS-Triton,
check out our complete [usage guide](https://github.com/rapidsai/rapids-triton/blob/main/docs/usage.md).

## Contributing

If you wish to contribute to RAPIDS-Triton, please see our [contributors'
guide](https://github.com/rapidsai/rapids-triton/blob/main/CONTRIBUTING.md) for
tips and full details on how to get started.
