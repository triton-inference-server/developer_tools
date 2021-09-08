<!--
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# The RAPIDS-Triton Linear Example

This repository offers an annotated example of how to create a custom Triton
backend using the RAPIDS-Triton library. In the following, we will demonstrate
step-by-step how to create a backend with RAPIDS-Triton that, when given two
vectors (**u** and **v**) as input will return a vector **r** according to the
following equation:

**r** = \alpha * **u** + **v** + **c**

where \alpha is a scalar constant read from a configuration file and **c** is a
constant vector read from a "model" file. Along the way, we will illustrate a
variety of useful operations in RAPIDS-Triton, including retrieving data from a
configuration file and loading model resources.

## 1. Getting Started

It is strongly recommended that you start from the [RAPIDS-Triton template
repo](https://github.com/rapidsai/rapids-triton-template) whenever you begin
developing a custom backend. This repo provides the boilerplate that would
otherwise be necessary to get started with your custom backend. Go ahead and
[create a new
repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/creating-a-repository-from-a-template) based on this template.

## 2. Pick a Name

Your custom backend will need a name that uniquely identifies it to the Triton
server. For this example, we will use the name `rapids_linear`. We will need to
provide this name in two places:


### 2.1 Update `names.h`
In the `src/names.h` file, adjust the definition of NAMESPACE to read

```cpp
#define NAMESPACE rapids_linear
```

Triton conventions require that backend definition be placed in a namespace of
the form `triton::backend::NAME_OF_BACKEND`, so we define the namespace name
here and use it where required in the `rapids-triton-template` code.

### 2.2 Update `CMakeLists.txt`
Near the top of `CMakeLists.txt` in the section labeled "Target names", there
is an option to provide a `BACKEND_NAME`. In this case, we will set this as
follows:

```cmake
set(BACKEND_NAME "rapids_linear")
```

## 2. Create an Example Configuration

[Configuration
files](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md)
provide settings for model deployments which determine how
the model will behave throughout its entire deployment (as opposed to values
which vary on a request-by-request basis). It is often helpful to think through
what your configuration file will look like before actually writing any code
for your custom backend. In the present example, we need to specify a few
different things:

1. The name of the backend which will be used to serve this model (the name
   chosen in step 2)
2. The names of the input vectors
3. The value of \alpha
4. Where to load **c** from

With this in mind, an example configuration file for the `rapids_linear`
backend might look something like this:

```protobuf
name: "linear_example"
backend: "rapids_linear"
max_batch_size: 32768
input [
  {
    name: "u"
    data_type: TYPE_FP32
    dims: [ 4 ]
  },
  {
    name: "v"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }
]
output [
  {
    name: "r"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }
]
instance_group [{ kind: KIND_GPU }]
parameters [
  {
    key: "alpha"
    value: { string_value: "2.0" }
  }
]
dynamic_batching {
  max_queue_delay_microseconds: 100
}
```

Let's review the pieces of this configuration file in a bit more detail, since
they will introduce several important concepts for development of our backend.

### `name`
This provides a name for an individual model. Remember that a backend is used
to serve a particular *kind* of model, but Triton can serve many different
models of this kind. The name given here specifies the specific model being
served in order to allow clients to submit requests to that model.

### `backend`
This is the name of the custom backend we are developing, in this case
`rapids_linear`.

### `max_batch_size`
All RAPIDS-Triton backends require that models specify some maximum batch size,
although this value can be arbitrarily large.

### `input`
This section specifies the input tensors used for this backend. Each input
tensor has a name (which we will use later in the actual custom backend code),
a
[datatype](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#datatypes)
(we use 32-bit floats in this example), and a shape. In this case, the **u**
and **v** vectors must be of the same shape as one another and as **c**, so we
will arbitrarily choose to make them of dimension 4 for our example
configuration.

### `output`
This section follows the same conventions as `input`. Again, for this
particular example, we must match the dimensions of the inputs and **c**.

### `instance_group`
This section  determines the hardware on which this model can be deployed (GPU
or CPU). For this example backend, we will demonstrate how to ensure that
models can perform inference on both GPUs and CPUs, buy we will start with GPU
inference.

### `parameters`
This section allows us to specify any other settings that may be required for
our model. Note that we specify all settings as `string_value` entries, but
RAPIDS-Triton will allow us to easily parse those values into bools, floats,
ints, or any other basic type.

### `dynamic_batching`
This section specifies how Triton's dynamic batcher will be used to combine
requests into batches for this model. By setting `max_queue_delay_microseconds`
to 100, we are allowing Triton to gather requests in a window of up to 100
microseconds before passing them to the backend. Triton's [model
analyzer](https://github.com/triton-inference-server/model_analyzer) can help
determine the optimal value for this window, but we will arbitrarily set it to
100 for this example.

Note that batching can happen both client-side and server-side. If a client
were to submit a request with 5x4 input tensors, for instance, the Triton
server will correctly interpret this as 5 different **u** and **v** vectors. It
can then batch this request with the next, which may carry 7x4 inputs or 1x4
inputs or any other valid mini-batch shape.

## 4. Define RapidsSharedState

Triton allows multiple copies or "instances" of a model to be loaded at the
same time. One obvious use case for this feature is to load the same model on
multiple GPUs, but it can also be used to "oversubscribe" hardware when this is
beneficial for meeting e.g. throughput or latency requirements. When multiple
instances of a model are loaded, they all have access to an object which
manages state that is relevant to all instances.

The most basic state that is shared among instances of a model is the model
configuration itself (as defined in step 3), but we can also use it to share
data that is specifically required by our backend. In our particular case, it
would be useful to cache the value of \alpha so that we do not have to retrieve
it from the configuration each time (which may involve additional parsing).

All RAPIDS-Triton backends store their shared state in a class called
`RapidsSharedState` in their own namespace, which inherits from
`rapids::SharedModelState`. A basic implementation of this class for the
current backend might look something like the following:

```cpp
struct RapidsSharedState : rapids::SharedModelState {
  RapidsSharedState(std::unique_ptr<common::TritonJson::Value>&& config)
      : rapids::SharedModelState{std::move(config)} {}
  void load() {}
  void unload() {}

  float alpha = 1.0f;
};
```

You may safely ignore the constructor in this example. It is boilerplate code
that should be included for all `RapidsSharedState` definitions and will not
change between backends.

Note that we have added two public member variables to this class definition
which will be used to store \alpha and **c**. One could equally well have made
these private members with getter functions or added arbitrarily complex logic
to this class definition, but we will leave them as is for simplicity.

### Accessing configuration parameters

Next, we need to actually load values into these newly-defined members. We can
do this by filling out the logic for our `load` method. For example, in order
to load \alpha, we could implement something like:
```cpp
void load() { alpha = get_config_param<float>("alpha"); }
```

Here, the `get_config_param` function allows us to directly access the `alpha`
parameter we defined in our example configuration file. By invoking the `float`
instantiation of this template, we ensure that we will retrieve the value as a
float.

### Unloading resources

In this particular case, our shared state does not include any resources that
need to be unloaded, but the `unload` method is available to do precisely that.
We will instead use it here simply to illustrate the use of RAPIDS-Triton
logging functions. Logging functions are defined in
`rapids_triton/triton/logging.hpp` and may be invoked as follows:
```cpp
void unload() { log_info(__FILE__, __LINE__) << "Unloading shared state..."; }
```
The arguments to `log_info` and associated functions may be omitted, but if
included may provide some use in debugging.

### Accessing model serialization files
Most backends will also need to deserialize their models from some file on-disk.
In our case, we are using the **c** vector as a stand-in for some more
interesting model representation. 
