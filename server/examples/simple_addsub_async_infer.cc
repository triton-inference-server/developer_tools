// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <unistd.h>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include "server_wrapper.h"


#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace tsw = triton::server::wrapper;

namespace {

#define FAIL(MSG)                                 \
  do {                                            \
    std::cerr << "error: " << (MSG) << std::endl; \
    exit(1);                                      \
  } while (false)
#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    triton::server::wrapper::Error err = (X);                      \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }
#ifdef TRITON_ENABLE_GPU
#define FAIL_IF_CUDA_ERR(X, MSG)                                           \
  do {                                                                     \
    cudaError_t err__ = (X);                                               \
    if (err__ != cudaSuccess) {                                            \
      std::cerr << "error: " << (MSG) << ": " << cudaGetErrorString(err__) \
                << std::endl;                                              \
      exit(1);                                                             \
    }                                                                      \
  } while (false)
#endif  // TRITON_ENABLE_GPU

bool enforce_memory_type = false;
tsw::Wrapper_MemoryType requested_memory_type;

#ifdef TRITON_ENABLE_GPU
static auto cuda_data_deleter = [](void* data) {
  if (data != nullptr) {
    cudaPointerAttributes attr;
    auto cuerr = cudaPointerGetAttributes(&attr, data);
    if (cuerr != cudaSuccess) {
      std::cerr << "error: failed to get CUDA pointer attribute of " << data
                << ": " << cudaGetErrorString(cuerr) << std::endl;
    }
    if (attr.type == cudaMemoryTypeDevice) {
      cuerr = cudaFree(data);
    } else if (attr.type == cudaMemoryTypeHost) {
      cuerr = cudaFreeHost(data);
    }
    if (cuerr != cudaSuccess) {
      std::cerr << "error: failed to release CUDA pointer " << data << ": "
                << cudaGetErrorString(cuerr) << std::endl;
    }
  }
};
#endif  // TRITON_ENABLE_GPU

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-m <\"system\"|\"pinned\"|gpu>"
            << " Enforce the memory type for input and output tensors."
            << " If not specified, inputs will be in system memory and outputs"
            << " will be based on the model's preferred type." << std::endl;
  std::cerr << "\t-v Enable verbose logging" << std::endl;

  exit(1);
}

template <typename T>
void
GenerateInputData(
    std::vector<char>* input0_data, std::vector<char>* input1_data)
{
  input0_data->resize(16 * sizeof(T));
  input1_data->resize(16 * sizeof(T));
  for (size_t i = 0; i < 16; ++i) {
    ((T*)input0_data->data())[i] = i;
    ((T*)input1_data->data())[i] = 1;
  }
}

template <typename T>
void
CompareResult(
    const std::string& output0_name, const std::string& output1_name,
    const void* input0, const void* input1, const char* output0,
    const char* output1)
{
  for (size_t i = 0; i < 16; ++i) {
    std::cout << ((T*)input0)[i] << " + " << ((T*)input1)[i] << " = "
              << ((T*)output0)[i] << std::endl;
    std::cout << ((T*)input0)[i] << " - " << ((T*)input1)[i] << " = "
              << ((T*)output1)[i] << std::endl;

    if ((((T*)input0)[i] + ((T*)input1)[i]) != ((T*)output0)[i]) {
      FAIL("incorrect sum in " + output0_name);
    }
    if ((((T*)input0)[i] - ((T*)input1)[i]) != ((T*)output1)[i]) {
      FAIL("incorrect difference in " + output1_name);
    }
  }
}

tsw::Error
ResponseAllocator(
    const char* tensor_name, size_t byte_size,
    tsw::Wrapper_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, tsw::Wrapper_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  std::cout << "Using custom allocation function" << std::endl;

  // Initially attempt to make the actual memory type and id that we
  // allocate be the same as preferred memory type
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    std::cout << "allocated " << byte_size << " bytes for result tensor "
              << tensor_name << std::endl;
  } else {
    void* allocated_ptr = nullptr;
    if (enforce_memory_type) {
      *actual_memory_type = requested_memory_type;
    }

    switch (*actual_memory_type) {
#ifdef TRITON_ENABLE_GPU
      case tsw::Wrapper_MemoryType::CPU_PINNED: {
        auto err = cudaSetDevice(*actual_memory_type_id);
        if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
            (err != cudaErrorInsufficientDriver)) {
          return triton::server::wrapper::Error(std::string(
              "unable to recover current CUDA device: " +
              std::string(cudaGetErrorString(err))));
        }

        err = cudaHostAlloc(&allocated_ptr, byte_size, cudaHostAllocPortable);
        if (err != cudaSuccess) {
          return triton::server::wrapper::Error(std::string(
              "cudaHostAlloc failed: " + std::string(cudaGetErrorString(err))));
        }
        break;
      }

      case tsw::Wrapper_MemoryType::GPU: {
        auto err = cudaSetDevice(*actual_memory_type_id);
        if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
            (err != cudaErrorInsufficientDriver)) {
          return triton::server::wrapper::Error(std::string(
              "unable to recover current CUDA device: " +
              std::string(cudaGetErrorString(err))));
        }

        err = cudaMalloc(&allocated_ptr, byte_size);
        if (err != cudaSuccess) {
          return triton::server::wrapper::Error(std::string(
              "cudaMalloc failed: " + std::string(cudaGetErrorString(err))));
        }
        break;
      }
#endif  // TRITON_ENABLE_GPU

      // Use CPU memory if the requested memory type is unknown
      // (default case).
      case tsw::Wrapper_MemoryType::CPU:
      default: {
        *actual_memory_type = tsw::Wrapper_MemoryType::CPU;
        allocated_ptr = malloc(byte_size);
        break;
      }
    }

    // Pass the tensor name with buffer_userp so we can show it when
    // releasing the buffer.
    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      *buffer_userp = new std::string(tensor_name);
      std::cout << "allocated " << byte_size << " bytes in "
                << WrapperMemoryTypeString(*actual_memory_type)
                << " for result tensor " << tensor_name << std::endl;
    }
  }

  return tsw::Error::Success;
}

tsw::Error
ResponseRelease(
    void* buffer, void* buffer_userp, size_t byte_size,
    tsw::Wrapper_MemoryType memory_type, int64_t memory_type_id)
{
  std::cout << "Using custom response release function" << std::endl;

  std::string* name = nullptr;
  if (buffer_userp != nullptr) {
    name = reinterpret_cast<std::string*>(buffer_userp);
  } else {
    name = new std::string("<unknown>");
  }

  std::stringstream ss;
  ss << buffer;
  std::string buffer_str = ss.str();

  std::cout << "Releasing buffer " << buffer_str << " of size "
            << std::to_string(byte_size) << " in "
            << tsw::WrapperMemoryTypeString(memory_type) << " for result '"
            << *name << std::endl;

  switch (memory_type) {
    case tsw::Wrapper_MemoryType::CPU:
      free(buffer);
      break;
#ifdef TRITON_ENABLE_GPU
    case tsw::Wrapper_MemoryType::CPU_PINNED: {
      auto err = cudaSetDevice(memory_type_id);
      if (err == cudaSuccess) {
        err = cudaFreeHost(buffer);
      }
      if (err != cudaSuccess) {
        std::cerr << "error: failed to cudaFree " << buffer << ": "
                  << cudaGetErrorString(err) << std::endl;
      }
      break;
    }
    case tsw::Wrapper_MemoryType::GPU: {
      auto err = cudaSetDevice(memory_type_id);
      if (err == cudaSuccess) {
        err = cudaFree(buffer);
      }
      if (err != cudaSuccess) {
        std::cerr << "error: failed to cudaFree " << buffer << ": "
                  << cudaGetErrorString(err) << std::endl;
      }
      break;
    }
#endif  // TRITON_ENABLE_GPU
    default:
      std::cerr << "error: unexpected buffer allocated in CUDA managed memory"
                << std::endl;
      break;
  }

  delete name;

  return tsw::Error::Success;
}

void
Check(
    tsw::Tensor& output0, tsw::Tensor& output1,
    const std::vector<char>& input0_data, const std::vector<char>& input1_data,
    const std::string& output0_name, const std::string& output1_name,
    const size_t expected_byte_size,
    const tsw::Wrapper_DataType expected_datatype,
    const std::string& model_name)
{
  std::unordered_map<std::string, std::vector<char>> output_data;
  for (auto& output : {output0, output1}) {
    if ((output.name_ != output0_name) && (output.name_ != output1_name)) {
      FAIL("unexpected output '" + output.name_ + "'");
    }

    if (model_name == "add_sub") {
      if ((output.shape_.size() != 1) || (output.shape_[0] != 16)) {
        FAIL("unexpected shape for '" + output.name_ + "'");
      }
    } else if (model_name == "simple") {
      if ((output.shape_.size() != 2) || (output.shape_[0] != 1) ||
          (output.shape_[1] != 16)) {
        FAIL("unexpected shape for '" + output.name_ + "'");
      }
    } else {
      FAIL("unexpected model name '" + model_name + "'");
    }

    if (output.data_type_ != expected_datatype) {
      FAIL(
          "unexpected datatype '" +
          std::string(WrapperDataTypeString(output.data_type_)) + "' for '" +
          output.name_ + "'");
    }

    if (output.byte_size_ != expected_byte_size) {
      FAIL(
          "unexpected byte-size, expected " +
          std::to_string(expected_byte_size) + ", got " +
          std::to_string(output.byte_size_) + " for " + output.name_);
    }

    // For the first infer request on 'add_sub' model, we use default allocator
    // so the memory type should be 'CPU'.
    if (model_name == "add_sub") {
      if (output.memory_type_ != tsw::Wrapper_MemoryType::CPU) {
        FAIL(
            "unexpected memory type, expected to be allocated in CPU, got " +
            std::string(WrapperMemoryTypeString(output.memory_type_)) +
            ", id " + std::to_string(output.memory_type_id_) + " for " +
            output.name_);
      }
    } else if (
        enforce_memory_type && (output.memory_type_ != requested_memory_type)) {
      FAIL(
          "unexpected memory type, expected to be allocated in " +
          std::string(WrapperMemoryTypeString(requested_memory_type)) +
          ", got " + std::string(WrapperMemoryTypeString(output.memory_type_)) +
          ", id " + std::to_string(output.memory_type_id_) + " for " +
          output.name_);
    }

    // We make a copy of the data here... which we could avoid for
    // performance reasons but ok for this simple example.
    std::vector<char>& odata = output_data[output.name_];
    switch (output.memory_type_) {
      case tsw::Wrapper_MemoryType::CPU: {
        std::cout << output.name_ << " is stored in system memory" << std::endl;
        odata.assign(output.buffer_, output.buffer_ + output.byte_size_);
        break;
      }

      case tsw::Wrapper_MemoryType::CPU_PINNED: {
        std::cout << output.name_ << " is stored in pinned memory" << std::endl;
        odata.assign(output.buffer_, output.buffer_ + output.byte_size_);
        break;
      }

#ifdef TRITON_ENABLE_GPU
      case tsw::Wrapper_MemoryType::GPU: {
        std::cout << output.name_ << " is stored in GPU memory" << std::endl;
        odata.reserve(output.byte_size_);
        FAIL_IF_CUDA_ERR(
            cudaMemcpy(
                &odata[0], output.buffer_, output.byte_size_,
                cudaMemcpyDeviceToHost),
            "getting " + output.name_ + " data from GPU memory");
        break;
      }
#endif

      default:
        FAIL("unexpected memory type");
    }
  }

  CompareResult<int32_t>(
      output0_name, output1_name, &input0_data[0], &input1_data[0],
      output_data[output0_name].data(), output_data[output1_name].data());
}

}  // namespace

int
main(int argc, char** argv)
{
  int verbose_level = 0;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vm:r:")) != -1) {
    switch (opt) {
      case 'm': {
        enforce_memory_type = true;
        if (!strcmp(optarg, "system")) {
          requested_memory_type = tsw::Wrapper_MemoryType::CPU;
        } else if (!strcmp(optarg, "pinned")) {
          requested_memory_type = tsw::Wrapper_MemoryType::CPU_PINNED;
        } else if (!strcmp(optarg, "gpu")) {
          requested_memory_type = tsw::Wrapper_MemoryType::GPU;
        } else {
          Usage(
              argv,
              "-m must be used to specify one of the following types:"
              " <\"system\"|\"pinned\"|gpu>");
        }
        break;
      }
      case 'v':
        verbose_level = 1;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

#ifndef TRITON_ENABLE_GPU
  if (enforce_memory_type && requested_memory_type != TRITONSERVER_MEMORY_CPU) {
    Usage(argv, "-m can only be set to \"system\" without enabling GPU");
  }
#endif  // TRITON_ENABLE_GPU


  // Use 'ServerOptions' object to initialize TritonServer. Here we set model
  // control mode to 'EXPLICIT' so that we are able to load and unload models
  // after startup.
  tsw::ServerOptions options({"./models"});
  options.logging_.verbose_ = verbose_level;
  options.model_control_mode_ =
      tsw::Wrapper_ModelControlMode::MODEL_CONTROL_EXPLICIT;
  auto server = tsw::TritonServer::Create(options);

  // Load 'simple' and 'add_sub' models.
  FAIL_IF_ERR(server->LoadModel("simple"), "loading 'simple' model");
  FAIL_IF_ERR(server->LoadModel("add_sub"), "loading 'add_sub' model");

  // Use 'ModelIndex' function to see model repository contents. Here we should
  // see both 'simple' and 'add_sub' models are ready.
  std::vector<tsw::RepositoryIndex> repo_index;
  FAIL_IF_ERR(server->ModelIndex(repo_index), "getting repository index");
  std::cout << "ModelIndex:\n";
  for (size_t i = 0; i < repo_index.size(); i++) {
    std::cout << repo_index[i].name_ << ", " << repo_index[i].version_ << ", "
              << repo_index[i].state_ << "\n";
  }

  // Initialize 'InferRequest' with the name of the model that we want to run an
  // inference on.
  auto request = tsw::InferRequest(tsw::InferOptions("add_sub"));

  // Add two input tensors to the inference request.
  std::vector<char> input0_data;
  std::vector<char> input1_data;
  GenerateInputData<int32_t>(&input0_data, &input1_data);
  size_t input0_size = input0_data.size();
  size_t input1_size = input1_data.size();

  // Use the iterator of input vector to add input data to a request.
  FAIL_IF_ERR(
      request.AddInput(
          "INPUT0", input0_data.begin(), input0_data.end(),
          tsw::Wrapper_DataType::INT32, {16}, tsw::Wrapper_MemoryType::CPU, 0),
      "adding input0");
  FAIL_IF_ERR(
      request.AddInput(
          "INPUT1", input1_data.begin(), input1_data.end(),
          tsw::Wrapper_DataType::INT32, {16}, tsw::Wrapper_MemoryType::CPU, 0),
      "adding input0");

  // Indicate that we want both output tensors calculated and returned
  // for the inference request. These calls are optional, if no
  // output(s) are specifically requested then all outputs defined by
  // the model will be calculated and returned.
  tsw::Tensor output0("OUTPUT0");
  tsw::Tensor output1("OUTPUT1");
  FAIL_IF_ERR(request.AddOutput(output0), "adding output0");
  FAIL_IF_ERR(request.AddOutput(output1), "adding output1");

  // Call 'AsyncInfer' function to run inference.
  std::future<tsw::InferResult> result_future;
  FAIL_IF_ERR(
      server->AsyncInfer(&result_future, request),
      "running the first async inference");

  // Get the infer result and check the result.
  auto result = result_future.get();
  if (result.HasError()) {
    FAIL(result.ErrorMsg());
  } else {
    std::string name = result.ModelName();
    std::string version = result.ModelVersion();
    std::string id = result.Id();
    std::cout << "Run inference on model '" << name << "', version '" << version
              << "', with request ID '" << id << "'\n";

    // Retrieve two outputs from the 'InferResult' object.
    FAIL_IF_ERR(result.Output(&output0), "getting result of output0");
    FAIL_IF_ERR(result.Output(&output1), "getting result of output1");

    Check(
        output0, output1, input0_data, input1_data, "OUTPUT0", "OUTPUT1",
        input0_size, tsw::Wrapper_DataType::INT32, result.ModelName());

    // Get full response.
    std::string debug_str;
    FAIL_IF_ERR(result.DebugString(&debug_str), "getting debug string");
    std::cout << debug_str << std::endl;
  }

  // Unload 'add_sub' model as we don't need it anymore.
  FAIL_IF_ERR(server->UnloadModel("add_sub"), "unloading 'add_sub' model");

  // Run a new infer requset on 'simple' model.
  request = tsw::InferRequest(tsw::InferOptions("simple"));

  // We can also use 'Tensor' object for adding input to a request.
  tsw::Tensor input0(
      "INPUT0", &input0_data[0], input0_data.size(),
      tsw::Wrapper_DataType::INT32, {1, 16}, tsw::Wrapper_MemoryType::CPU, 0);
  tsw::Tensor input1(
      "INPUT1", &input1_data[0], input1_data.size(),
      tsw::Wrapper_DataType::INT32, {1, 16}, tsw::Wrapper_MemoryType::CPU, 0);
  FAIL_IF_ERR(request.AddInput(input0), "adding input0");
  FAIL_IF_ERR(request.AddInput(input1), "adding input1");

  // For this inference, we provide pre-allocated buffer for output. The infer
  // result will be stored in-place to the buffer.
  void* allocated_output0 = malloc(64);
  void* allocated_output1 = malloc(64);
  tsw::Tensor alloc_output0(
      "OUTPUT0", reinterpret_cast<char*>(allocated_output0), 64);
  tsw::Tensor alloc_output1(
      "OUTPUT1", reinterpret_cast<char*>(allocated_output1), 64);
  FAIL_IF_ERR(request.AddOutput(alloc_output0), "adding alloc_output0");
  FAIL_IF_ERR(request.AddOutput(alloc_output1), "adding alloc_output1");

  // Call 'AsyncInfer' function to run inference. Here we need to pass a future
  // of 'ErrorCheck' object as an argument that we use for waiting until the
  // result in the pre-allocated buffer is ready and checking if an error occurs
  // during the inference.
  std::future<tsw::ErrorCheck> err_check;
  FAIL_IF_ERR(
      server->AsyncInfer(&err_check, request),
      "running the second async inference");

  auto error = err_check.get();
  if (error.has_error_) {
    FAIL(error.error_message_);
  } else {
    // Check the output data in the pre-allocated buffer.
    CompareResult<int32_t>(
        "OUTPUT0", "OUTPUT1", &input0_data[0], &input1_data[0],
        reinterpret_cast<const char*>(allocated_output0),
        reinterpret_cast<const char*>(allocated_output1));
  }
  // Need to free the provided buffer.
  free(allocated_output0);
  free(allocated_output1);

  // Clear all the responses that have been completed before using custom
  // allocator below.
  tsw::ClearCompletedResponses();

  // For the third inference, we use custom allocator for output allocation.
  // Initialize the allocator with our custom functions 'ResponseAllocator' and
  // 'ResponseRelease' which are implemented above.
  tsw::Allocator allocator(ResponseAllocator, ResponseRelease);
  auto infer_options = tsw::InferOptions("simple");
  infer_options.custom_allocator_ = &allocator;
  request = tsw::InferRequest(infer_options);

  const void* input0_base = &input0_data[0];
  const void* input1_base = &input1_data[0];
#ifdef TRITON_ENABLE_GPU
  std::unique_ptr<void, decltype(cuda_data_deleter)> input0_gpu(
      nullptr, cuda_data_deleter);
  std::unique_ptr<void, decltype(cuda_data_deleter)> input1_gpu(
      nullptr, cuda_data_deleter);
  bool use_cuda_memory =
      (enforce_memory_type &&
       (requested_memory_type != tsw::Wrapper_MemoryType::CPU));
  if (use_cuda_memory) {
    FAIL_IF_CUDA_ERR(cudaSetDevice(0), "setting CUDA device to device 0");
    if (requested_memory_type != tsw::Wrapper_MemoryType::CPU_PINNED) {
      void* dst;
      FAIL_IF_CUDA_ERR(
          cudaMalloc(&dst, input0_size),
          "allocating GPU memory for INPUT0 data");
      input0_gpu.reset(dst);
      FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, &input0_data[0], input0_size, cudaMemcpyHostToDevice),
          "setting INPUT0 data in GPU memory");
      FAIL_IF_CUDA_ERR(
          cudaMalloc(&dst, input1_size),
          "allocating GPU memory for INPUT1 data");
      input1_gpu.reset(dst);
      FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, &input1_data[0], input1_size, cudaMemcpyHostToDevice),
          "setting INPUT1 data in GPU memory");
    } else {
      void* dst;
      FAIL_IF_CUDA_ERR(
          cudaHostAlloc(&dst, input0_size, cudaHostAllocPortable),
          "allocating pinned memory for INPUT0 data");
      input0_gpu.reset(dst);
      FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, &input0_data[0], input0_size, cudaMemcpyHostToHost),
          "setting INPUT0 data in pinned memory");
      FAIL_IF_CUDA_ERR(
          cudaHostAlloc(&dst, input1_size, cudaHostAllocPortable),
          "allocating pinned memory for INPUT1 data");
      input1_gpu.reset(dst);
      FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, &input1_data[0], input1_size, cudaMemcpyHostToHost),
          "setting INPUT1 data in pinned memory");
    }
  }

  input0_base = use_cuda_memory ? input0_gpu.get() : &input0_data[0];
  input1_base = use_cuda_memory ? input1_gpu.get() : &input1_data[0];
#endif  // TRITON_ENABLE_GPU

  // Reuse the two inputs and modify the buffer and memory type based on the
  // commandline.
  input0.buffer_ = reinterpret_cast<char*>(const_cast<void*>(input0_base));
  input1.buffer_ = reinterpret_cast<char*>(const_cast<void*>(input1_base));
  input0.memory_type_ = requested_memory_type;
  input1.memory_type_ = requested_memory_type;

  FAIL_IF_ERR(request.AddInput(input0), "adding input0");
  FAIL_IF_ERR(request.AddInput(input1), "adding input1");

  // Call 'AsyncInfer' function to run inference.
  FAIL_IF_ERR(
      server->AsyncInfer(&result_future, request),
      "running the third async inference");

  // Get the infer result and check the result.
  result = result_future.get();
  if (result.HasError()) {
    FAIL(result.ErrorMsg());
  } else {
    std::string name = result.ModelName();
    std::string version = result.ModelVersion();
    std::string id = result.Id();
    std::cout << "Run inference on model '" << name << "', version '" << version
              << "', with request ID '" << id << "'\n";

    // Retrieve two outputs from the 'InferResult' object.
    FAIL_IF_ERR(result.Output(&output0), "getting result of output0");
    FAIL_IF_ERR(result.Output(&output1), "getting result of output1");

    Check(
        output0, output1, input0_data, input1_data, "OUTPUT0", "OUTPUT1",
        input0_size, tsw::Wrapper_DataType::INT32, result.ModelName());

    // Get full response.
    std::string debug_str;
    FAIL_IF_ERR(result.DebugString(&debug_str), "getting debug string");
    std::cout << debug_str << std::endl;
  }

  // Get the server metrics.
  std::string metrics_str;
  FAIL_IF_ERR(server->Metrics(&metrics_str), "fetching metrics");
  std::cout << "=========Metrics===========\n" << metrics_str << "\n";

  return 0;
}
