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
[Triton in-process C-API](https://github.com/triton-inference-server/server/blob/main/docs/inference_protocols.md#in-process-triton-server-api)
, providing a more simple interface for users to use Triton in-process C API for
developing their application without having in-depth knowledge of Triton
implementation details or writing complicated code. This wrapper is also called
the "Higher Level In Process C++ API" or just "Server Wrapper" for short. The
header file that defines and documents the Server C-API Wrapper is
[server_wrapper.h](include/server_wrapper.h). Ask questions or report problems
in the main Triton
[issues page](https://github.com/triton-inference-server/server/issues).

## Build the Server C-API Wrapper library and custom application 

To build and install the Server Wrapper library use the following commands.

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

When the build completes, the library `libtritonserverwrapper.a` and examples
can be found in the install directory.

For custom application, you can refer to
[CMakeLists.txt](examples/CMakelists.txt) to see how to build your executable
with the Server Wrapper library.

### API Description

Triton Server C-API Wrapper is encapsulated in a shared library which is built
from source contained in this repository. You can include the full
capabilities by linking the shared library into your application and by using
the C++ API defined in [server_wrapper.h](include/server_wrapper.h).

#### Inference APIs

Three main objects will be used for Server Wrapper.

##### TritonServer

The top-level abstraction used by Server Wrapper is `TritonServer`,
which represents the Triton core logic that is capable of implementing
some of the features and capabilities of Triton.

##### InferRequest

`InferRequest` carries the information for a inference request. This object
allows you to set inference options, add inputs and outputs to a request.

##### InferResult

`InferResult` provides an interface to interpret the inference response, making
it more easily to retrieve output data.

##### General Workflow

Performing an inference request requires the use of some Server C++ API
functions and objects, as demonstrated in
[simple_addsub_async_infer.cc](./server/examples/simple_addsub_async_infer.cc).
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
[model control mode](https://github.com/triton-inference-server/server/blob/main/docs/model_management.md)
is set to "EXPLICIT" when setting the server options in the previous step, you
can load a specific model by calling

```cpp
server.LoadModel("your_model_name");
```

3. Construct `InferRequest` with infer options

Initialize the request with `InferOptions` structure, specifying the name of
the model that you want to run an inference on and other inference options.

```cpp
auto request = InferRequest(InferOptions("your_model_name"));
```

4. Add inputs / requested outputs to a request

You can add input to a request by either using `Tensor` object, which contains
the information of an input tensor, or using the iterator if the input data is
stored in a container. Iterator can also be used if input data is a container
holding 'string' elements. For output, you can add the name of requested
output to a request, indicating what output to be calculated and returned for
inference. You can also provide pre-allocated buffer for output if you want
the output data to be stored in-place in the provided buffer.  Please see
[simple_addsub_async_infer.cc](./server/examples/simple_addsub_async_infer.cc)
for detailed usage.

```cpp
Tensor input0(.....);
Tensor input1(.....);
request.AddInput(input0);
request.AddInput(input1);

Tensor output0("OUTPUT0_NAME");
Tensor output1("OUTPUT1_NAME");
request.AddOutput(output0);
request.AddOutput(output1);
```

5. Call the inference method

Server Wrapper uses promise-future based structure for asynchronous inference.
A future will be passed as an argument to `AsyncInfer` function so that the
result can be retrieved whenever needed by calling `future.get()`.

When running inference, Server Wrapper provides three options for the
allocation for output tensors.

* Use default allocator

Default output allocation and release functions will be used by default.

```cpp
std::future<InferResult> result_future;
server->AsyncInfer(&result_future, request);
auto result = result_future.get();

// Retrieve output data from 'InferResult' object...
...
```

* Use custom allocator

You can provide your custom allocator using `Allocator` object. You need to
register your callback functions to the allocator when creating the
`Allocator` object, and set `InferOptions` properly when initializing
`InferRequest`. The signatures of the callback functions are defined in
[server_wrapper.h](include/server_wrapper.h).

```cpp
// 'ResponseAllocator' and 'ResponseRelease' are the custom output allocation and release functions.
Allocator allocator(ResponseAllocator, ResponseRelease);
auto infer_options = InferOptions("your_model_name");
infer_options.custom_allocator_ = &allocator;
auto request = InferRequest(infer_options);

// Call the inference method, and the custom allocator will be used.
std::future<InferResult> result_future;
server->AsyncInfer(&result_future, request);
auto result = result_future.get();

// Retrieve output data from 'InferResult' object...
...
```

* Use pre-allocated buffer

You can also pre-allocate buffers for output tensors. The output data will be
stored in the buffer you provided when adding outputs to a request in the
previous step. Note that those buffers should be freed when they are no longer
needed.

```cpp
...
void* buffer_ptr0 = malloc(64);
void* buffer_ptr1 = malloc(64);
Tensor output0("OUTPUT0_NAME", buffer_ptr0, ...);
Tensor output1("OUTPUT1_NAME", buffer_ptr1, ...);
request.AddOutput(output0);
request.AddOutput(output1);

// Pass a future of 'ErrorCheck' object for checking if error occurs.
std::future<ErrorCheck> err_check;
server->AsyncInfer(&err_check, request);
auto error = err_check.get();

// Retrieve data from buffer...
...

free(buffer_ptr0);
free(buffer_ptr1);
```

#### Non-Inference APIs

Server Wrapper contains APIs for loading/unloading models, getting metrics an
model index, etc. The use of these functions is straightforward and these
functions are demonstrated in
[simple_addsub_async_infer.cc](./server/examples/simple_addsub_async_infer.cc)
and
[addsub_string_async_infer.cc](./server/examples/addsub_string_async_infer.cc),
and documented in [server_wrapper.h](include/server_wrapper.h).

#### Error Handling

Most Higher Level Server C++ API functions return an error object indicating
success or failure. Success is indicated by return `Error::Success` to
indicate no error. Failure is indicated by returning a `Error` object with the
error message. To check the error status and message, use `Error.IsOk()` and
`Error.Message()` with the `Error` object.


A simple example using the Server Wrapper can be found in
[simple_addsub_async_infer.cc](./server/examples/simple_addsub_async_infer.cc)
which is heavily commented. For string type IO, an example can be found in
[addsub_string_async_infer.cc](./server/examples/addsub_string_async_infer.cc).

#### Note

Currently, Server Wrapper only provides limited functionality of Triton Server
C-API. For example, decoupled models are not supported. More features will be
supported in the future.
