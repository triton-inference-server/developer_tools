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

# Using RAPIDS-Triton

## Getting Started
To begin developing a custom backend with RAPIDS-Triton, we strongly recommend
that you take advantage of the [rapids-triton-template
repo](https://github.com/rapidsai/rapids-triton-template), which provides a
basic template for your backend code. If this is your first time developing a
backend with RAPIDS-Triton, the easiest way to get started is to follow the
[Linear Example](https://github.com/rapidsai/rapids-triton-linear-example).
This provides a detailed walkthrough of every step in the process of creating a
backend with example code. The rest of these usage docs will provide general
information on specific features you are likely to take advantage of when
building your backend.

## Logging
To provide logging messages in your backend, RAPIDS-Triton provides `log_info`,
`log_warn`, `log_error`, and `log_debug`. During default Triton execution, all
logging messages up to (but not including) debug level will be visible. These
functions can be invoked in two ways and can optionally include file and line
information. To add a logging message to your code, use one of the following
invocations:
```cpp
#include <rapids_triton/triton/logging.hpp>

void logging_example() {
  rapids::log_info() << "This is a log message.";
  rapids::log_info("This is an equivalent invocation.");
  rapids::log_info(__FILE__, __LINE__) << "This one has file and line info.";
  rapids::log_info(__FILE__, __LINE__, "And so does this one.");
}
```

## Error Handling
If you encounter an error condition at any point in your backend which cannot
be otherwise handled, you should throw a `TritonException`. In most cases, this
error will be gracefully handled and passed to the Triton server in a way that
will not interfere with execution of other backends, models, or requests.

`TritonException` objects are constructed with an error type and a message
indicating what went wrong, as shown below:
```cpp
#include <rapids_triton/exceptions.hpp>

void error_example() {
  throw rapids::TritonException(rapids::Error::Internal, "Something bad happened!");
}
```

Available error types are:
* `Internal`: The most common error type. Used when an unexpected condition
  which is not the result of bad user input (e.g. CUDA error).
* `NotFound`: An error type returned when a named resource (e.g. named CUDA IPC
  memory block) cannot be found.
* `InvalidArg`: An error type returned when the user has provided invalid input
  in a configuration file or request.
* `Unavailable`: An error returned when a resource exists but is currently
  unavailable.
* `Unsupported`: An error which indicates that a requested functionality is not
  implemented by this backend (e.g. GPU execution for a CPU-only backend).
* `AlreadyExists`: An error which indicates that a resource which is being
  created has already been created.
* `Unknown`: The type of the error cannot be established. This type should be
  avoided wherever possible.

The `cuda_check` function is provided to help facilitate error handling of
direct invocations of the CUDA API. If such an invocation fails, `cuda_check`
will throw an appropriate `TritonException`:

```cpp
#include <rapids_triton/exceptions.hpp>

void cuda_check_example() {
  rapids::cuda_check(cudaSetDevice(0));
}
```

If a `TritonException` is thrown while a backend is being loaded, Triton's
server logs will indicate the failure and include the error message. If a
`TritonException` is thrown while a model is being loaded, Triton's server logs
will display the error message in the loading logs for that model. If a
`TritonException` is thrown during handling of a request, the client will
receive an indication that the request failed along with the error message, but
the model can continue to process other requests.

## CPU-Only Builds
Most Triton backends include support for builds intended to support only CPU
execution. While this is not required, RAPIDS-Triton includes a compile-time
constant which can be useful for facilitating this:

```cpp
#include <rapids_triton/build_control.hpp>

void do_a_gpu_thing() {
  if constexpr (rapids::IS_GPU_BUILD) {
    rapids::log_info("Executing on GPU...");
  } else {
    rapids::log_error("Can't do that! This is a CPU-only build.");
  }
}
```

You can also make use of the preprocessor identifier `TRITON_ENABLE_GPU` for
conditional inclusion of headers:
```cpp
#ifdef TRITON_ENABLE_GPU
#include <gpu_stuff.h>
#endif
```

## Buffers
Within a backend, it is often useful to process data in a way that is agnostic
to whether the underlying memory is on the host or on device and whether that
memory is owned by the backend or provided by Triton. For instance, a backend
may receive input data from Triton on the host and conditionally transfer it to
the GPU before processing. In this case, owned memory must be allocated on the
GPU to store the data, but after that point, the backend will treat the data
exactly the same as if Triton had provided it on device in the first place.

In order to handle such situations, RAPIDS-Triton provides the `Buffer` object.
When the `Buffer` is non-owning, it provides a lightweight wrapper to the
underlying memory. When it is owning, `Buffer` will handle any necessary
deallocation (on host or device). These objects can also be extremely useful
for passing data back and forth between host and device. The following examples
show ways in which `Buffer` objects can be constructed and used:

```cpp
#include <utility>
#include <vector>
#include <rapids_triton/memory/types.hpp>  // rapids::HostMemory and rapids::DeviceMemory
#include <rapids_triton/memory/buffer.hpp> // rapids::Buffer

void buffer_examples() {
  auto data = std::vector<int>{0, 1, 2, 3, 4};

  // This buffer is a lightweight wrapper around the data stored in the `data`
  // vector. Because this constructor takes an `int*` pointer as its first
  // argument, it is assumed that the lifecycle of the underlying memory is
  // separately managed.
  auto non_owning_host_buffer = rapids::Buffer<int>(data.data(), rapids::HostMemory);

  // This buffer owns its own memory on the host, with space for 5 ints. When
  // it goes out of scope, the memory will be appropriately deallocated.
  auto owning_host_buffer = rapids::Buffer<int>(5, rapids::HostMemory);

  // This buffer is constructed as a copy of `non_owning_host_buffer`. Because
  // its requested memory type is `DeviceMemory`, the data will be copied to a
  // new (owned) GPU allocation. Device and stream can also be specified in the
  // constructor.
  auto owning_device_buffer = rapids::Buffer<int>(non_owning_host_buffer, rapids::DeviceMemory);

  // Once again, because this constructor takes an `int*` pointer, it will
  // simply be a lightweight wrapper around the memory that is actually managed
  // by `owning_device_buffer`. Here we have omitted the memory type argument,
  // since it defaults to `DeviceMemory`. This constructor can also accept
  // device and stream arguments, and care should be taken to ensure that the
  // right device is specified when the buffer does not allocate its own
  // memory.
  auto non_owning_device_buffer = rapids::Buffer<int>(owning_host_buffer.data());

  auto base_buffer1 = rapids::Buffer<int>(data.data(), rapids::HostMemory);
  // Because this buffer is on the host, just like the (moved-from) buffer it
  // is being constructed from, it remains non-owning
  auto non_owning_moved_buffer = rapids::Buffer<int>(std::move(base_buffer1), rapids::HostMemory);

  auto base_buffer2 = rapids::Buffer<int>(data.data(), rapids::HostMemory);
  // Because this buffer is on the device, unlike the (moved-from) buffer it is
  // being constructed from, memory must be allocated on-device, and the new
  // buffer becomes owning.
  auto owning_moved_buffer = rapids::Buffer<int>(std::move(base_buffer2), rapids::DeviceMemory);
}
```

### Useful Methods
* `data()`: Return a raw pointer to the buffer's data
* `size()`: Return the number of elements contained by the buffer
* `mem_type()`: Return the type of memory (`HostMemory` or `DeviceMemory`)
  contained by the buffer
* `device()`: Return the id of the device on which this buffer resides (always
  0 for host buffers)
* `stream()`: Return the CUDA stream associated with this buffer.
* `stream_synchronize()`: Perform a stream synchronization on this buffer's
  stream.
* `set_stream(cudaStream_t new_stream)`: Synchronize on the current stream and
  then switch buffer to the new stream.

## Tensors
`Tensor` objects are wrappers around `Buffers` with some additional metadata
and functionality. All `Tensor` objects have a shape which can be retrieved as
a `std::vector` using the `shape()` method. A reference to the underlying
buffer can also be retrieved with the `buffer()` method.

`OutputTensor` objects are used to store data which will eventually be returned
as part of Triton's response to a client request. Their `finalize` methods are
used to actually marshal their underlying data into a response.

In general, `OutputTensor` objects should not be constructed directly but
should instead be retrieved using the `get_output` method of a `Model`
(described later).

## Moving Data: `rapids::copy`
Moving data around between host and device or simply between buffers of the
same type can be one of the more error-prone tasks outside of actual model
execution in a backend. To help make this process easier, RAPIDS-Triton
provides a number of overrides of the `rapids::copy` function, which provides a
safe way to mode data between buffers or tensors. Assuming the size attribute
of the buffer or tensor has not been corrupted, `rapids::copy` should never
result in segfaults or invalid memory access on device.

Additional overrides of `rapids::copy` exist, but we will describe the most
common uses of it here. Note that you need not worry about where the underlying
data is located (on host or device) when invoking `rapids::copy`. The function
will take care of detecting and handling this. `Tensor` overrides are in
`rapids_triton/tensor/tensor.hpp` and `Buffer` overrides are in
`rapids_triton/memory/buffer.hpp`.

### Between two buffers or tensors...
If you wish to simply copy the entire contents of one buffer into another or
one tensor into another, `rapids::copy` can be invoked as follows:
```cpp
rapids::copy(destination_buffer, source_buffer);
rapids::copy(destination_tensor, source_tensor);
```
If the destination is too small to contain the data from the source, a
`TritonException` will be thrown.

### From one tensor to many...
To distribute data from one tensor to many, the following override is
available:
```cpp
rapids::copy(iterator_to_first_destination, iterator_to_last_destination, source);
```
Note that destination tensors can be of different sizes. If the destination
buffers cannot contain all data from the  source, a `TritonException` will be
thrown. Destination tensors can also be a mixture of device and host tensors if
desired.

### From part of one buffer to part of another...
To move data from part of one buffer to part of another, you can use another
override as in the following example:
```cpp
rapids::copy(destination_buffer, source_buffer, 10, 3, 6);
```
The extra arguments here provide the offset from the beginning of the
destination buffer to which data should be copied, the index of the beginning
element to be copied from the source, and the index one past the final element to
be copied from the source. Thus, this invocation will copy the third, fourth,
and fifth elements of the source buffer to the tenth, eleventh, and twelfth
elements of the destination. If the destination buffer only had room for (e.g.)
eleven elements, a `TritonException` would be thrown.

## `Model`
For a thorough introduction to developing a RAPIDS-Triton `Model` for your
backend, see the [Linear Example
repo](https://github.com/rapidsai/rapids-triton-linear-example). Here, we will
just briefly summarize some of the useful methods of `Model` objects.

### Non-Virtual Methods
* `get_input`: Used to retrieve an input tensor of a particular name from
  Triton
* `get_output`: Used to retrieve an output tensor of a particular name from
  Triton
* `get_config_param`: Used to retrieve a named parameter from the configuration
  file for this model
* `get_device_id`: The device on which this model is deployed (0 for host
  deployments)
* `get_deployment_type`: One of `GPUDeployment` or `CPUDeployment` depending on
  whether this model is configured to be deployed on device or host

### Virtual Methods
* `predict`: The method which performs actual inference on input data and
  stores it to the output location
* `load`: A method which can be overridden to load resources that will be used
  for the lifetime of the model
* `unload`: A method used to unload any resources loaded in `load` if
  necessary
* `preferred_mem_type`, `preferred_mem_type_in`, and `preferred_mem_type_out`:
  The location (device or host) where input and output data should be stored.
  The latter two methods can be overridden if input and output data should be
  stored differently. Otherwise, `preferred_mem_type` will be used for both.
* `get_stream`: A method which can be overridden to provide different streams
  for handling successive batches. Otherwise, the default stream associated
  with this model will be used.

## `SharedState`
Multiple instances of a RAPIDS-Triton model may need to share some data between
them (or may choose to do so for efficiency). `SharedState` objects facilitate
this.  For a thorough introduction to developing a RAPIDS-Triton `SharedState`
for your backend, see the [Linear Example
repo](https://github.com/rapidsai/rapids-triton-linear-example). Just like the
`Model` objects which share a particular `SharedState` object, configuration
parameters can be retrieved using `SharedState`'s `get_config_param` method.
Otherwise, most additional functionality is defined by the backend
implementation, including `load` and `unload` methods for any necessary
loading/unloading of resources that will be used for the lifetime of the shared
state.

Note that just one shared state is constructed by the server regardless of how
many instances of a given model are created.

## Other Memory Allocations
For most device memory allocations, it is strongly recommended that you simply
construct a `Buffer` of the correct size and type. However, if you do not wish
to use a `Buffer` in a particular context, you are encouraged to allocate and
deallocate device memory using [RMM](https://github.com/rapidsai/rmm). Any
memory managed in this way will make use of Triton's CUDA memory pool, which
will be faster than performing individual allocations. Memory can be allocated
and deallocated using RMM as follows:

```cpp
#include <rmm/mr/device/per_device_resource.hpp>

void rmm_example(std::size_t number_of_bytes, cudaStream_t stream) {
  void* data = rmm::get_current_device_resource()->allocate(number_of_bytes, stream);
  rmm::get_current_device_resource()->deallocate(data, number_of_bytes);
}
```

It is strongly recommended that you not change the RMM device resource in your
backend, since doing so will cause allocations to no longer make use of
Triton's memory pool.
