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
#include "triton/developer_tools/server_wrapper.h"


#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace tds = triton::developer_tools::server;

namespace {

#define FAIL(MSG)                                 \
  do {                                            \
    std::cerr << "error: " << (MSG) << std::endl; \
    exit(1);                                      \
  } while (false)
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
tds::MemoryType requested_memory_type;

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

void
ResponseAllocator(
    const char* tensor_name, size_t byte_size,
    tds::MemoryType preferred_memory_type, int64_t preferred_memory_type_id,
    void** buffer, tds::MemoryType* actual_memory_type,
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
    std::cout << "allocated " << byte_size << " bytes for result tensor "
              << tensor_name << std::endl;
  } else {
    void* allocated_ptr = nullptr;
    if (enforce_memory_type) {
      *actual_memory_type = requested_memory_type;
    }

    switch (*actual_memory_type) {
#ifdef TRITON_ENABLE_GPU
      case tds::MemoryType::CPU_PINNED: {
        auto err = cudaSetDevice(*actual_memory_type_id);
        if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
            (err != cudaErrorInsufficientDriver)) {
          throw tds::TritonException(std::string(
              "unable to recover current CUDA device: " +
              std::string(cudaGetErrorString(err))));
        }

        err = cudaHostAlloc(&allocated_ptr, byte_size, cudaHostAllocPortable);
        if (err != cudaSuccess) {
          throw tds::TritonException(std::string(
              "cudaHostAlloc failed: " + std::string(cudaGetErrorString(err))));
        }
        break;
      }

      case tds::MemoryType::GPU: {
        auto err = cudaSetDevice(*actual_memory_type_id);
        if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
            (err != cudaErrorInsufficientDriver)) {
          throw tds::TritonException(std::string(
              "unable to recover current CUDA device: " +
              std::string(cudaGetErrorString(err))));
        }

        err = cudaMalloc(&allocated_ptr, byte_size);
        if (err != cudaSuccess) {
          throw tds::TritonException(std::string(
              "cudaMalloc failed: " + std::string(cudaGetErrorString(err))));
        }
        break;
      }
#endif  // TRITON_ENABLE_GPU

      // Use CPU memory if the requested memory type is unknown
      // (default case).
      case tds::MemoryType::CPU:
      default: {
        *actual_memory_type = tds::MemoryType::CPU;
        allocated_ptr = malloc(byte_size);
        break;
      }
    }

    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      std::cout << "allocated " << byte_size << " bytes in "
                << MemoryTypeString(*actual_memory_type)
                << " for result tensor " << tensor_name << std::endl;
    }
  }
}

void
ResponseRelease(
    void* buffer, size_t byte_size, tds::MemoryType memory_type,
    int64_t memory_type_id)
{
  std::cout << "Using custom response release function" << std::endl;

  std::stringstream ss;
  ss << buffer;
  std::string buffer_str = ss.str();

  std::cout << "Releasing buffer " << buffer_str << " of size "
            << std::to_string(byte_size) << " in "
            << tds::MemoryTypeString(memory_type);

  switch (memory_type) {
    case tds::MemoryType::CPU:
      free(buffer);
      break;
#ifdef TRITON_ENABLE_GPU
    case tds::MemoryType::CPU_PINNED: {
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
    case tds::MemoryType::GPU: {
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
}

void
Check(
    std::shared_ptr<tds::Tensor>& output0,
    std::shared_ptr<tds::Tensor>& output1, const std::vector<char>& input0_data,
    const std::vector<char>& input1_data, const std::string& output0_name,
    const std::string& output1_name, const size_t expected_byte_size,
    const tds::DataType expected_datatype, const std::string& model_name,
    const bool is_custom_alloc)
{
  std::unordered_map<std::string, std::vector<char>> output_data;
  for (auto& output : {std::make_pair(output0_name, output0),
                       std::make_pair(output1_name, output1)}) {
    if (model_name == "add_sub") {
      if ((output.second->shape_.size() != 1) ||
          (output.second->shape_[0] != 16)) {
        FAIL("unexpected shape for '" + output.first + "'");
      }
    } else if (model_name == "simple") {
      if ((output.second->shape_.size() != 2) ||
          (output.second->shape_[0] != 1) || (output.second->shape_[1] != 16)) {
        FAIL("unexpected shape for '" + output.first + "'");
      }
    } else {
      FAIL("unexpected model name '" + model_name + "'");
    }

    if (output.second->data_type_ != expected_datatype) {
      FAIL(
          "unexpected datatype '" +
          std::string(DataTypeString(output.second->data_type_)) + "' for '" +
          output.first + "'");
    }

    if (output.second->byte_size_ != expected_byte_size) {
      FAIL(
          "unexpected byte-size, expected " +
          std::to_string(expected_byte_size) + ", got " +
          std::to_string(output.second->byte_size_) + " for " + output.first);
    }

    // For this example, we use default allocator and pre-allocated buffer in
    // the first and second infer requests, so the memory type for both cases
    // should be 'CPU'.
    if (is_custom_alloc) {
      if (enforce_memory_type &&
          (output.second->memory_type_ != requested_memory_type)) {
        FAIL(
            "unexpected memory type, expected to be allocated in " +
            std::string(MemoryTypeString(requested_memory_type)) + ", got " +
            std::string(MemoryTypeString(output.second->memory_type_)) +
            ", id " + std::to_string(output.second->memory_type_id_) + " for " +
            output.first);
      }
    } else {
      if (output.second->memory_type_ != tds::MemoryType::CPU) {
        FAIL(
            "unexpected memory type, expected to be allocated in CPU, got " +
            std::string(MemoryTypeString(output.second->memory_type_)) +
            ", id " + std::to_string(output.second->memory_type_id_) + " for " +
            output.first);
      }
    }

    // We make a copy of the data here... which we could avoid for
    // performance reasons but ok for this simple example.
    std::vector<char>& odata = output_data[output.first];
    switch (output.second->memory_type_) {
      case tds::MemoryType::CPU: {
        std::cout << output.first << " is stored in system memory" << std::endl;
        odata.assign(
            output.second->buffer_,
            output.second->buffer_ + output.second->byte_size_);
        break;
      }

      case tds::MemoryType::CPU_PINNED: {
        std::cout << output.first << " is stored in pinned memory" << std::endl;
        odata.assign(
            output.second->buffer_,
            output.second->buffer_ + output.second->byte_size_);
        break;
      }

#ifdef TRITON_ENABLE_GPU
      case tds::MemoryType::GPU: {
        std::cout << output.first << " is stored in GPU memory" << std::endl;
        odata.reserve(output.second->byte_size_);
        FAIL_IF_CUDA_ERR(
            cudaMemcpy(
                &odata[0], output.second->buffer_, output.second->byte_size_,
                cudaMemcpyDeviceToHost),
            "getting " + output.first + " data from GPU memory");
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
          requested_memory_type = tds::MemoryType::CPU;
        } else if (!strcmp(optarg, "pinned")) {
          requested_memory_type = tds::MemoryType::CPU_PINNED;
        } else if (!strcmp(optarg, "gpu")) {
          requested_memory_type = tds::MemoryType::GPU;
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

  try {
    // Use 'ServerOptions' object to initialize TritonServer. Here we set model
    // control mode to 'EXPLICIT' so that we are able to load and unload models
    // after startup.
    tds::ServerOptions options({"./models"});
    options.logging_.verbose_ =
        tds::LoggingOptions::VerboseLevel(verbose_level);
    options.model_control_mode_ = tds::ModelControlMode::EXPLICIT;
    auto server = tds::TritonServer::Create(options);

    // Load 'simple' and 'add_sub' models.
    server->LoadModel("simple");
    server->LoadModel("add_sub");
    // Use 'ModelIndex' function to see model repository contents. Here we
    // should see both 'simple' and 'add_sub' models are ready.
    std::vector<tds::RepositoryIndex> repo_index = server->ModelIndex();
    std::cout << "ModelIndex:\n";
    for (size_t i = 0; i < repo_index.size(); i++) {
      std::cout << repo_index[i].name_ << ", " << repo_index[i].version_ << ", "
                << ModelReadyStateString(repo_index[i].state_) << "\n";
    }

    // Initialize 'InferRequest' with the name of the model that we want to run
    // an inference on.
    auto request1 = tds::InferRequest::Create(tds::InferOptions("add_sub"));

    // Add two input tensors to the inference request.
    std::vector<char> input0_data;
    std::vector<char> input1_data;
    GenerateInputData<int32_t>(&input0_data, &input1_data);
    size_t input0_size = input0_data.size();
    size_t input1_size = input1_data.size();

    // Use the iterator of input vector to add input data to a request.
    request1->AddInput(
        "INPUT0", input0_data.begin(), input0_data.end(), tds::DataType::INT32,
        {16}, tds::MemoryType::CPU, 0);
    request1->AddInput(
        "INPUT1", input1_data.begin(), input1_data.end(), tds::DataType::INT32,
        {16}, tds::MemoryType::CPU, 0);

    // Indicate that we want both output tensors calculated and returned
    // for the inference request. These calls are optional, if no
    // output(s) are specifically requested then all outputs defined by
    // the model will be calculated and returned.
    request1->AddRequestedOutput("OUTPUT0");
    request1->AddRequestedOutput("OUTPUT1");

    // Call 'AsyncInfer' function to run inference.
    auto result_future1 = server->AsyncInfer(*request1);

    // Get the infer result and check the result.
    auto result1 = result_future1.get();
    if (result1->HasError()) {
      FAIL(result1->ErrorMsg());
    }
    std::cout << "Ran inference on model '" << result1->ModelName()
              << "', version '" << result1->ModelVersion()
              << "', with request ID '" << result1->Id() << "'\n";

    // Retrieve two outputs from the 'InferResult' object.
    std::shared_ptr<tds::Tensor> result1_out0 = result1->Output("OUTPUT0");
    std::shared_ptr<tds::Tensor> result1_out1 = result1->Output("OUTPUT1");

    Check(
        result1_out0, result1_out1, input0_data, input1_data, "OUTPUT0",
        "OUTPUT1", input0_size, tds::DataType::INT32, result1->ModelName(),
        false);

    // Get full response.
    std::cout << result1->DebugString() << std::endl;


    // Unload 'add_sub' model as we don't need it anymore.
    server->UnloadModel("add_sub");
    // Run a new infer requset on 'simple' model.
    auto request2 = tds::InferRequest::Create(tds::InferOptions("simple"));

    // We can also use 'Tensor' object for adding input to a request.
    tds::Tensor input0(
        &input0_data[0], input0_data.size(), tds::DataType::INT32, {1, 16},
        tds::MemoryType::CPU, 0);
    tds::Tensor input1(
        &input1_data[0], input1_data.size(), tds::DataType::INT32, {1, 16},
        tds::MemoryType::CPU, 0);
    request2->AddInput("INPUT0", input0);
    request2->AddInput("INPUT1", input1);

    // For this inference, we provide pre-allocated buffer for output. The infer
    // result will be stored in-place to the buffer.
    std::shared_ptr<void> allocated_output0(malloc(64), free);
    std::shared_ptr<void> allocated_output1(malloc(64), free);

    tds::Tensor alloc_output0(
        reinterpret_cast<char*>(allocated_output0.get()), 64,
        tds::MemoryType::CPU, 0);
    tds::Tensor alloc_output1(
        reinterpret_cast<char*>(allocated_output1.get()), 64,
        tds::MemoryType::CPU, 0);
    request2->AddRequestedOutput("OUTPUT0", alloc_output0);
    request2->AddRequestedOutput("OUTPUT1", alloc_output1);

    // Call 'AsyncInfer' function to run inference.
    auto result_future2 = server->AsyncInfer(*request2);

    // Get the infer result and check the result.
    auto result2 = result_future2.get();
    if (result2->HasError()) {
      FAIL(result2->ErrorMsg());
    }
    std::cout << "Ran inference on model '" << result2->ModelName()
              << "', version '" << result2->ModelVersion()
              << "', with request ID '" << result2->Id() << "'\n";

    // Retrieve two outputs from the 'InferResult' object.
    std::shared_ptr<tds::Tensor> result2_out0 = result2->Output("OUTPUT0");
    std::shared_ptr<tds::Tensor> result2_out1 = result2->Output("OUTPUT1");

    Check(
        result2_out0, result2_out1, input0_data, input1_data, "OUTPUT0",
        "OUTPUT1", input0_size, tds::DataType::INT32, result2->ModelName(),
        false);

    // Get full response.
    std::cout << result2->DebugString() << std::endl;

    // Check the output data in the pre-allocated buffer.
    CompareResult<int32_t>(
        "OUTPUT0", "OUTPUT1", &input0_data[0], &input1_data[0],
        reinterpret_cast<const char*>(allocated_output0.get()),
        reinterpret_cast<const char*>(allocated_output1.get()));

    // For the third inference, we use custom allocator for output allocation.
    // Initialize the allocator with our custom functions 'ResponseAllocator'
    // and 'ResponseRelease' which are implemented above.
    std::shared_ptr<tds::Allocator> allocator(
        new tds::Allocator(ResponseAllocator, ResponseRelease));
    auto infer_options = tds::InferOptions("simple");
    infer_options.custom_allocator_ = allocator;
    auto request3 = tds::InferRequest::Create(infer_options);

    const void* input0_base = &input0_data[0];
    const void* input1_base = &input1_data[0];
#ifdef TRITON_ENABLE_GPU
    std::unique_ptr<void, decltype(cuda_data_deleter)> input0_gpu(
        nullptr, cuda_data_deleter);
    std::unique_ptr<void, decltype(cuda_data_deleter)> input1_gpu(
        nullptr, cuda_data_deleter);
    bool use_cuda_memory =
        (enforce_memory_type &&
         (requested_memory_type != tds::MemoryType::CPU));
    if (use_cuda_memory) {
      FAIL_IF_CUDA_ERR(cudaSetDevice(0), "setting CUDA device to device 0");
      if (requested_memory_type != tds::MemoryType::CPU_PINNED) {
        void* dst;
        FAIL_IF_CUDA_ERR(
            cudaMalloc(&dst, input0_size),
            "allocating GPU memory for INPUT0 data");
        input0_gpu.reset(dst);
        FAIL_IF_CUDA_ERR(
            cudaMemcpy(
                dst, &input0_data[0], input0_size, cudaMemcpyHostToDevice),
            "setting INPUT0 data in GPU memory");
        FAIL_IF_CUDA_ERR(
            cudaMalloc(&dst, input1_size),
            "allocating GPU memory for INPUT1 data");
        input1_gpu.reset(dst);
        FAIL_IF_CUDA_ERR(
            cudaMemcpy(
                dst, &input1_data[0], input1_size, cudaMemcpyHostToDevice),
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

    request3->AddInput("INPUT0", input0);
    request3->AddInput("INPUT1", input1);

    // Call 'AsyncInfer' function to run inference.
    auto result_future3 = server->AsyncInfer(*request3);

    // Get the infer result and check the result.
    auto result3 = result_future3.get();
    if (result3->HasError()) {
      FAIL(result3->ErrorMsg());
    }
    std::cout << "Ran inference on model '" << result3->ModelName()
              << "', version '" << result3->ModelVersion()
              << "', with request ID '" << result3->Id() << "'\n";

    // Retrieve two outputs from the 'InferResult' object.
    std::shared_ptr<tds::Tensor> result3_out0 = result3->Output("OUTPUT0");
    std::shared_ptr<tds::Tensor> result3_out1 = result3->Output("OUTPUT1");

    Check(
        result3_out0, result3_out1, input0_data, input1_data, "OUTPUT0",
        "OUTPUT1", input0_size, tds::DataType::INT32, result3->ModelName(),
        true);

    // Get full response.
    std::cout << result3->DebugString() << std::endl;

    // Get the server metrics.
    std::string metrics_str = server->ServerMetrics();
    std::cout << "\n\n\n=========Server Metrics===========\n" << metrics_str << "\n";
  }
  catch (const tds::TritonException& ex) {
    std::cerr << "Error: " << ex.what();
    exit(1);
  }

  return 0;
}
