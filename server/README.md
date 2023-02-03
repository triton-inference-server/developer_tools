<!--
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Triton Server C-API Wrapper

Triton Server C-API Wrapper wraps up the functionality of
[Triton in-process C-API](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inference_protocols.md#in-process-triton-server-api)
, providing a simpler interface for users to use Triton in-process C API for
developing their application without having in-depth knowledge of Triton
implementation details or writing complicated code. This wrapper is also called
the "Higher Level In Process C++ API" or just "Server Wrapper" for short. The
header file that defines and documents the Server C-API Wrapper is
[server_wrapper.h](include/triton/developer_tools/server_wrapper.h). Ask
questions or report problems in the main Triton
[issues page](https://github.com/triton-inference-server/server/issues).

## Build the Server C-API Wrapper library and custom application 

To build and install the Server Wrapper library from
`developer_tools/server`, use the following commands.

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]
* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]

See the [CMakeLists.txt](CMakeLists.txt) file for other build options.

When the build completes, the library `libtritondevelopertoolsserver.a` and examples
can be found in the install directory.

For custom application, you can refer to
[CMakeLists.txt](examples/CMakeLists.txt) to see how to build your executable
with the Server Wrapper library.

### API Description

Triton Server C-API Wrapper is encapsulated in a shared library which is built
from source contained in this repository. You can include the full
capabilities by linking the shared library into your application and by using
the C++ API defined in [server_wrapper.h](include/triton/developer_tools/server_wrapper.h).

#### Inference APIs

Three main objects will be used for Server Wrapper.

##### TritonServer

The top-level abstraction used by Server Wrapper is `TritonServer`,
which represents the Triton core logic that is capable of implementing
some of the features and capabilities of Triton.

##### InferRequest

`InferRequest` carries the information for a inference request. This object
allows you to set inference options, add inputs and requeseted outputs to a request.

##### InferResult

`InferResult` provides an interface to interpret the inference response, making
it more easily to retrieve output data.

##### General Workflow

Performing an inference request requires the use of some Server C++ API
functions and objects, as demonstrated in
[simple_addsub_async_infer.cc](examples/simple_addsub_async_infer.cc).
The general usage requires the following steps.

1. Start Server

To start a Triton server, you need to  create a `TritonServer` instance with
the `ServerOptions` structure which contains the server options used to
initialize the server.

```cpp
auto server = TritonServer::Create(ServerOptions options({"path/to/your/model_repository", "path/to/another/model_repository"}));
```

2. Load model (optional)

This step is optional as all the models in the model repository paths provided
in the previous step will be loaded to the server by default. However, if
[model control mode](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md)
is set to "EXPLICIT" when setting the server options in the previous step, you
can load a specific model by calling

```cpp
server->LoadModel("your_model_name");
```

3. Construct `InferRequest` with infer options

Initialize the request with `InferOptions` structure, specifying the name of
the model that you want to run an inference on and other inference options.

```cpp
auto request = InferRequest::Create(InferOptions("your_model_name"));
```

4. Add inputs / requested outputs to a request

You can add an input to a request by either using `Tensor` object, which
contains the information of an input tensor, or using the iterator if the input
data is stored in a contiguous container. Iterator can also be used if input
data is of 'string' type and is stored in a contiguous container. Note that the
input data buffer within the 'Tensor' object must not be modified until
inference is completed and result is returned.

For output, you can add the name of requested output to a request, indicating
what output to be calculated and returned for inference. You can also provide
pre-allocated buffer for output in this step if you want the output data to
be stored in-place in the provided buffer.  See "Use pre-allocated buffer"
section in the next step for more information.

```cpp
// Assume that we have input data in these two vectors.
std::vector<char> input0_data;
std::vector<char> input1_data;

Tensor input0(&input0_data[0], input0_data.size(), DataType::INT32, {1, 16}, MemoryType::CPU, 0);
Tensor input1(&input1_data[0], input1_data.size(), DataType::INT32, {1, 16}, MemoryType::CPU, 0);

request->AddInput("INPUT0_NAME", input0);
request->AddInput("INPUT1_NAME", input1);

request->AddRequestedOutput("OUTPUT0_NAME");
request->AddRequestedOutput("OUTPUT1_NAME");
```

5. Call the inference method

Server Wrapper uses promise-future based structure for asynchronous inference.
A future of a unique pointer of `InferResult` object will be returned from
`AsyncInfer` function, and the result can be retrieved whenever needed by
calling `future.get()`.

When running inference, Server Wrapper provides three options for the
allocation and deallocation of output tensors.

* Use default allocator

Default output allocation/deallocation will be used. No need to specify how to
allocate/deallocate the output tensors.

```cpp
// Call the inference method.
std::future<std::unique_ptr<InferResult>> result_future = server->AsyncInfer(*request);

// Get the infer result and check the result.
auto result = result_future.get();
if (result->HasError()) {
    std::cerr << result->ErrorMsg();
} else {
    // Retrieve output data from 'InferResult' object...
}
```

* Use custom allocator

You can provide your custom allocator using `Allocator` object. You need to
register your callback functions to the allocator when creating the
`Allocator` object, and set `InferOptions` properly when initializing
`InferRequest`. The signatures of the callback functions are defined in
[common.h](include/triton/developer_tools/common.h).

```cpp
// 'ResponseAllocator' and 'ResponseRelease' are the custom output allocation
// and deallocation functions.
Allocator allocator(ResponseAllocator, ResponseRelease);
auto infer_options = InferOptions("your_model_name");

// Set custom allocator to 'InferOptions'.
infer_options.custom_allocator_ = &allocator;
auto request = InferRequest(infer_options);

/**
Add inputs/requested outputs to a request as shown in the previous step...
*/

// Call the inference method, and the custom allocator will be used.
std::future<std::unique_ptr<InferResult>> result_future = server->AsyncInfer(*request);

// Get the infer result and check the result.
auto result = result_future.get();
if (result->HasError()) {
    std::cerr << result->ErrorMsg();
} else {
    // Retrieve output data from 'InferResult' object...
}
```

* Use pre-allocated buffer

You can pre-allocate buffers for output tensors. The output data will be
stored in the buffer you provided when adding requested outputs to a request in
the previous step. Note that those buffers will *not* be freed when the `Tensor`
object goes out of scope, and should be freed manually when they are no
longer needed.

```cpp
/*
Add inputs to a request as shown in the previous step...
*/

void* buffer_ptr0 = malloc(64);
void* buffer_ptr1 = malloc(64);

// Provide pre-allocated buffer for each output tensor.
Tensor output0(reinterpret_cast<char*>(buffer_ptr0), 64, MemoryType::CPU, 0);
Tensor output1(reinterpret_cast<char*>(buffer_ptr1), 64, MemoryType::CPU, 0);

request->AddRequestedOutput("OUTPUT0_NAME", output0);
request->AddRequestedOutput("OUTPUT1_NAME", output1);

// Call the inference method.
std::future<std::unique_ptr<InferResult>> result_future = server->AsyncInfer(*request);

// Get the infer result and check the result.
auto result = result_future.get();
if (result->HasError()) {
    std::cerr << result->ErrorMsg();
} else {
    // Retrieve output data from 'InferResult' object...
}

// Need to free the buffer manually.
free(buffer_ptr0);
free(buffer_ptr1);
```

The lifetime of output data is owned by each returned output `Tensor` object.
For cases using default allocator or custom allocator, the deallocation of
the buffer where the output data is stored will occurs when the `Tensor`
object goes out of scope.

#### Non-Inference APIs

Server Wrapper contains APIs for loading/unloading models, getting metrics, and
model index, etc. The use of these functions is straightforward and these
functions are documented in
[server_wrapper.h](include/triton/developer_tools/server_wrapper.h). You can
find some of the functions demonstrated in the [examples](examples).

#### Error Handling

Most Higher Level Server C++ API functions throws a `TritonException` when an
error occurs. You can utilize `TritonException`, which is documented in
[common.h](include/triton/developer_tools/common.h), in your application for
error handling.

#### Examples

A simple example using the Server Wrapper can be found in
[simple_addsub_async_infer.cc](examples/simple_addsub_async_infer.cc)
which is heavily commented. For string type IO, an example can be found in
[addsub_string_async_infer.cc](examples/addsub_string_async_infer.cc). For
decoupled models, please refer to
[square_async_infer.cc](examples/square_async_infer.cc).

When running the examples, make sure the model repository is placed under the
same path, and `LD_LIBRARY_PATH` is set properly for `libtritonserver.so`.

```
# Prepare the models required by the examples.

$ cd /path/to/developer_tools/server

$ mkdir -p ./examples/models

# Copy over the models placed in the qa folder.
$ cp -r ../qa/L0_server_unit_test/models/add_sub* ./examples/models/.

# Copy over the models placed in the server repository.
$ git clone https://github.com/triton-inference-server/server.git
$ cp -r server/docs/examples/model_repository/simple ./examples/models/.

# Copy over the decoupled model placed in the python_backend repository.
$ git clone https://github.com/triton-inference-server/python_backend.git
$ mkdir -p ./examples/models/square_int32/1
$ cp python_backend/examples/decoupled/square_model.py ./examples/models/square_int32/1/model.py
$ cp python_backend/examples/decoupled/square_config.pbtxt ./examples/models/square_int32/config.pbtxt

# Copy over the executables from the install directory.
$ cp /path/to/install/bin/simple_addsub_async_infer ./examples
$ cp /path/to/install/bin/addsub_string_async_infer ./examples
$ cp /path/to/install/bin/square_async_infer ./examples

# Assume libtritonserver.so is placed under "/opt/tritonserver/lib"
$ LD_LIBRARY_PATH=/opt/tritonserver/lib:${LD_LIBRARY_PATH}

$ cd ./examples

# Run examples
$ ./simple_addsub_async_infer
$ ./addsub_string_async_infer
$ ./square_async_infer
```
