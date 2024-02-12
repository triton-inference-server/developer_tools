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

**r** = &alpha; * **u** + **v** + **c**

where &alpha; is a scalar constant read from a configuration file and **c** is a
constant vector read from a "model" file. Along the way, we will illustrate a
variety of useful operations in RAPIDS-Triton, including retrieving data from a
configuration file and loading model resources.

This example is intended to provide a fair amount of depth about backend
development with RAPIDS-Triton. For a simpler example, check out the
pass-through backend in the main [RAPIDS-Triton
repo](https://github.com/rapidsai/rapids-triton#simple-example).
For even more detail on RAPIDS-Triton features introduced here, check out the
API documentation.

All of the following steps are written as if you were starting from scratch in
creating this backend, but you can also just browse the files in this repo to
see how the final version of the backend might look.

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

## 3. Create an Example Configuration

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
3. The value of &alpha;
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
would be useful to cache the value of &alpha; so that we do not have to retrieve
it from the configuration each time (which may involve additional parsing).

All RAPIDS-Triton backends store their shared state in a class called
`RapidsSharedState` in their own namespace, which inherits from
`dev_tools::SharedModelState`. A basic implementation of this class for the
current backend might look something like the following:

```cpp
struct RapidsSharedState : dev_tools::SharedModelState {
  RapidsSharedState(std::unique_ptr<common::TritonJson::Value>&& config)
      : dev_tools::SharedModelState{std::move(config)} {}
  void load() {}
  void unload() {}

  float alpha = 1.0f;
};
```

You may safely ignore the constructor in this example. It is boilerplate code
that should be included for all `RapidsSharedState` definitions and will not
change between backends. Take a look at `src/shared_state.h` to see this
implementation in context.

Note that we have added a public member variables to this class definition
which will be used to store &alpha;. One could equally well have made
these private members with getter functions or added arbitrarily complex logic
to this class definition, but we will leave them as is for simplicity.

### Accessing configuration parameters

Next, we need to actually load a value into this newly-defined member. We can
do this by filling out the logic for our `load` method. For example, in order
to load &alpha;, we could implement something like:
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
void unload() { dev_tools::log_info(__FILE__, __LINE__) << "Unloading shared state..."; }
```
The arguments to `log_info` and related functions may be omitted, but if
included may provide some use in debugging.

## 5. Define RapidsModel

The real heart of any RAPIDS-Triton backend is its RapidsModel definition. This
class implements the actual logic to deserialize a model and use it to perform
inference.

### 5.1 Deserialize the model
Most backends will need to deserialize their models from some file on-disk.
In our case, we are using the **c** vector as a stand-in for some more
interesting model structure. For simplicity, we will assume that **c** is
just stored as space-separated floats in a text file.

A natural question is why we did not perform this deserialization in
`RapidsSharedState::load` since **c** is the same for all instances of a model.
In general, instances may be initialized on different GPUs, or some may be on
GPUs while others are on CPUs, so we defer model deserialization until we know
where the model should be deserialized *to*.

Given that **c** could be stored on either host or device, the question of how
to represent it as a member variable becomes a little more fraught. We could
represent it as a raw pointer, but this requires a great deal of manual
tracking to determine whether to use `std::malloc` or `cudaMalloc` for the
initial allocation in the model `load()` method and `std::free` or `cudaFree`
in `unload()`.

To help simplify how situations like this are handled, RAPIDS-Triton introduces
a lightweight `Buffer` class, which can provide unified RAII access to memory
on host or device **or** a non-owning view on host/device memory that was
allocated elsewhere. We'll introduce a `Buffer` member to RapidsModel to store **c**:

```cpp
dev_tools::Buffer<float> c{};
```

Now, we are ready to actually load **c** from its file representation.
`dev_tools::Model`, the abstract parent class of `RapidsModel` implementations,
defines several methods to help with model deserialization, including:
- `get_filepath`, which returns the path to the model file if it is specified
  in the config or the model directory if it is not
- `get_deployment_type`, which returns whether this model should be deployed on
  CPU or GPU
- `get_device_id`, which returns the id of the device on which the model should
  be deployed (always 0 for CPU deployments)

Using these functions, we can define our `load()` function as follows. First,
we determine the full filepath to the model file, defaulting to a name of
`c.txt` if it is not specified in the config file:
```cpp
if (std::filesystem::is_directory(path)) {
  path /= "c.txt";
}
```

Next, we actually read the values of **c** into a temporary vector from its
text file representation:
```cpp
 auto model_vec = std::vector<float>{};
 auto model_file = std::ifstream(path.string());
 auto input_line = std::string{};
 std::getline(model_file, input_line);
 auto input_stream = std::stringstream{input_line};
 auto value = 0.0f;
 while (input_stream >> value) {
   model_vec.push_back(value);
 }
```

We then query the model to figure out exactly what sort of Buffer will be
needed to store **c**:
```cpp
auto memory_type = dev_tools::MemoryType{};
if (get_deployment_type() == dev_tools::GPUDeployment) {
  memory_type = dev_tools::DeviceMemory;
} else {
  memory_type = dev_tools::HostMemory;
}
c = dev_tools::Buffer<float>(model_vec.size(), memory_type, get_device_id());
```

Finally, we use the helper function `dev_tools::copy` to copy the values of **c**
from a Buffer-based view of `model_vec` to the `c` Buffer itself:
```cpp
dev_tools::copy(c, dev_tools::Buffer<float>(model_vec.data(), model_vec.size(),
                                      dev_tools::HostMemory));
```

By taking advantage of Buffer's RAII semantics, we eliminate the need to
explicitly implement an `unload` function, but we could do so if necessary.

### 5.2 Write the predict function
Now all that remains is to use our loaded "model" for inference. The `predict`
method of a RapidsModel implementation takes in a `Batch` object as argument,
which can be used to retrieve the input and output tensors that we will operate
on during inference. We can retrieve these tensors using the names we
originally specified in the config file:

```cpp
auto u = get_input<float>(batch, "u");
auto v = get_input<float>(batch, "v");
auto r = get_output<float>(batch, "r");
```

By default, the location of the returned tensors (host or device) is determined
by whether the model is deployed on host or device and whether or not the
backend was compiled with GPU-support enabled. You may choose to override the
`preferred_mem_type` method of your RapidsModel implementation in order to
specify a different general rule, or you can optionally pass a `MemoryType` to
`get_input` and `get_output` for even finer-grained control. Here, we will
simply accept the default behavior and use the `mem_type` method of the
returned tensor to determine how inference will proceed.

For tensors on the host, our inference logic might look something like:

```cpp
if (u.mem_type() == dev_tools::HostMemory) {
  auto alpha = get_shared_state()->alpha;
  for (std::size_t i{}; i < u.size(); ++i) {
    r.data()[i] =
        alpha * u.data()[i] + v.data()[i] + c.data()[i % c.size()];
  }
}
```

We'll define the logic for GPU inference in a separate `.cu` file like so:
```cuda
__global__ void cu_gpu_infer(float* r, float const* u, float const* v,
                             float* c, float alpha, std::size_t features,
                             std::size_t length) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < length) {
    r[id] = alpha * u[id] + v[id] + c[id % features];
  }
}

void gpu_infer(float* r, float const* u, float const* v, float* c, float alpha,
               std::size_t features, std::size_t length, cudaStream_t stream) {
  auto constexpr block_size = 1024;
  auto grid_size = static_cast<int>(std::max(1.0f, std::ceil(length /
          static_cast<float>(block_size))));;
  cu_gpu_infer<<<grid_size, block_size, 0, stream>>>(r, u, v, c, alpha, features, length);
}
```

and then call it within our RapidsModel `predict` method via:

```cpp
gpu_infer(r.data(), u.data(), v.data(), c.data(), alpha, c.size(),
        u.size(), r.stream());
```

After the actual inference has been performed, the one remaining task is to
call the `finalize` method of all output tensors. In this example, we have
exactly one, so the final line of our `predict` method is just:

```cpp
r.finalize();
```

To see all of this in context, check out the `src/gpu_infer.h` and
`src/gpu_infer.cu` files where GPU inference has been implemented as well as
`src/model.h` where it is used. When introducing a new source file, don't
forget to add it to CMakeLists.txt so that it will be included in the build.

## 6. Build the backend
Having defined all the necessary logic for serving our model, we can now
actually build the server container with the new backend included. To do so,
run the following command from the base of the repository:

```bash
docker build --build-arg BACKEND_NAME=rapids_linear -t rapids_linear .
```

## 7. Test

All that remains is to test that the backend performs correctly. In this repo,
we have provided two model directories in `qa/L0_e2e/model_repository`. These
models are identical except that one is deployed on CPU and the other on GPU.
The `config.pbtxt` files are laid out exactly as discussed in step 3.

To start the server with these models, run the following:

```bash
docker run \
  --gpus=all \
  --rm \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $PWD/qa/L0_e2e/model_repository:/models
  rapids_linear
```

You can now submit inference requests via any Triton client. For convenience,
we will use the client provided by the `rapids_triton` package in the main
RAPIDS-Triton repo. This package is primarily designed to assist with writing
end-to-end tests for RAPIDS-Triton backends. We can install it into a conda
environment as follows:

```bash
conda env create -f conda/environments/rapids_triton_test.yml
conda activate rapids_triton_test
python -m pip install git+https://github.com/rapidsai/rapids-triton.git#subdirectory=python
```

To use it for a basic test, we might execute something like the following

```python
from rapids_triton import Client

client = Client()

u = np.array([[2, 2, 3, 3]], dtype='float32')
v = np.array([[-1, 1, 1, -1]], dtype='float32')
# The value of the c vector specified in c.txt
c = np.array([[1, 2, 3, 4]], dtype='float32')
alpha = 2

ground_truth = alpha * u + v + np.repeat(c, u.shape[0], axis=0)

print({'r': ground_truth})

print(client.predict(
    # Specify name of model to use for prediction
    'linear_example',
    # Provide input arrays
    {'u': u, 'v': v},
    # Provide size in bytes of expected output(s)
    {'r': u.shape[0] * u.shape[1] * np.dtype('float32').itemsize},
    # Optionally submit request with Triton's shared memory mode
    shared_mem='cuda'
))
print(client.predict(
    'linear_example_cpu',
    {'u': u, 'v': v},
    {'r': u.shape[0] * u.shape[1] * np.dtype('float32').itemsize}
))
```
which should give us the following output:
```
{'r': array([[ 4.,  7., 10.,  9.]], dtype=float32)}
{'r': array([[ 4.,  7., 10.,  9.]], dtype=float32)}
{'r': array([[ 4.,  7., 10.,  9.]], dtype=float32)}
```

While this suggests that the backend is operating correctly, we probably want
to set up a more robust test with larger input data for use in CI and
development testing. See `qa/L0_e2e/test_model.py` for an example of how such a
test might be created.

## Conclusion
This walkthrough has provided an in-depth look at how to create a Triton
backend using RAPIDS-Triton, from initial description of the backend behavior
to end-to-end testing of models deployed using this backend. Following similar
steps, you should be able to integrate almost any algorithm for deployment with
Triton.

While we have tried to cover a wide variety of possible use cases with this
example, there is more to explore in the [RAPIDS-Triton
documentation](https://github.com/rapidsai/rapids-triton/blob/main/docs/usage.md)
itself.  If there is something you would like to do with RAPIDS-Triton which
does not seem to be covered by the available API or if something is not working
as expected, please submit a feature request or bug report to the
[RAPIDS-Triton issue
tracker](https://github.com/rapidsai/rapids-triton/issues). If you think this
example could be improved or expanded in some way, please [submit a pull
request](https://github.com/rapidsai/rapids-triton-linear-example/pulls) or
[issue](https://github.com/rapidsai/rapids-triton-linear-example/issues) to
this repo.

For additional information about using and deploying Triton after creating a
backend like this, check out the [main Triton
repo](https://github.com/triton-inference-server/server/). There you will find
information about many more tools to help you get the most out of Triton,
including:
- [`perf_analyzer`](https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md):
  A tool to help measure throughput and latency for models deployed with Triton
- [`model_analyzer`](https://github.com/triton-inference-server/server/blob/main/docs/model_analyzer.md):
  A tool to help determine what parameters will optimize throughput, latency,
  or any other metric for your deployed models
- [Helm
  charts](https://ngc.nvidia.com/catalog/helm-charts/nvidia:tritoninferenceserver/)
  and other information to help you easily deploy Triton in any cloud service
  or orchestration environment
