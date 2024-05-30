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
#include <chrono>

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
    std::vector<char>* input0_data)
{
  input0_data->resize(150528 * sizeof(T));
  for (size_t i = 0; i < 150528; ++i) {
    ((T*)input0_data->data())[i] = 1.0 * i;
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
  for (auto& output :
       {std::make_pair(output0_name, output0),
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
    // Enable tracing. The tracing output file 'trace_file' can be found after
    // this example is completed.
    options.trace_ = std::make_shared<tds::Trace>(
        "trace_file", tds::Trace::Level::TIMESTAMPS, 1, -1, 0);
    auto server = tds::TritonServer::Create(options);

    // Load resnet50 ONNX models.
    server->LoadModel("resnet50");

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
    auto request1 = tds::InferRequest::Create(tds::InferOptions("resnet50"));
    auto request2 = tds::InferRequest::Create(tds::InferOptions("resnet50"));
    auto request3 = tds::InferRequest::Create(tds::InferOptions("resnet50"));
  
    // Add two input tensors to the inference request.
    std::vector<char> input0_data;
    GenerateInputData<float>(&input0_data);
    size_t input0_size = input0_data.size();

    std::vector<char> input1_data;
    GenerateInputData<float>(&input1_data);
    size_t input1_size = input1_data.size();

    std::vector<char> input2_data;
    GenerateInputData<float>(&input2_data);
    size_t input2_size = input2_data.size();



    // Use the iterator of input vector to add input data to a request.
    request1->AddInput(
        "gpu_0/data_0", input0_data.begin(), input0_data.end(), tds::DataType::FP32,
        {3,224,224}, tds::MemoryType::CPU, 0);

    request2->AddInput(
        "gpu_0/data_0", input1_data.begin(), input1_data.end(), tds::DataType::FP32,
        {3,224,224}, tds::MemoryType::CPU, 0);
    
    request3->AddInput(
        "gpu_0/data_0", input2_data.begin(), input2_data.end(), tds::DataType::FP32,
        {3,224,224}, tds::MemoryType::CPU, 0);

    // For this inference, we provide pre-allocated buffer for output. The infer
    // result will be stored in-place to the buffer.
    std::shared_ptr<void> allocated_output0(malloc(4000), free);
    std::shared_ptr<void> allocated_output1(malloc(4000), free);
    std::shared_ptr<void> allocated_output2(malloc(4000), free);

    tds::Tensor alloc_output0(
        reinterpret_cast<char*>(allocated_output0.get()), 4000,
        tds::MemoryType::CPU, 0);
    tds::Tensor alloc_output1(
        reinterpret_cast<char*>(allocated_output1.get()), 4000,
        tds::MemoryType::CPU, 0);
    tds::Tensor alloc_output2(
        reinterpret_cast<char*>(allocated_output2.get()), 4000,
        tds::MemoryType::CPU, 0);

    request1->AddRequestedOutput("gpu_0/softmax_1", alloc_output0);
    request2->AddRequestedOutput("gpu_0/softmax_1", alloc_output1);
    request3->AddRequestedOutput("gpu_0/softmax_1", alloc_output2);

    auto start = std::chrono::high_resolution_clock::now();

    // Call 'AsyncInfer' function to run inference.
    auto result_future1 = server->AsyncInfer(*request1);

    // Get the infer result and check the result.
    auto result1 = result_future1.get();
    if (result1->HasError()) {
      FAIL(result1->ErrorMsg());
    }

    auto result_future2 = server->AsyncInfer(*request2);

    // Get the infer result and check the result.
    auto result2 = result_future2.get();
    if (result2->HasError()) {
      FAIL(result2->ErrorMsg());
    }

    auto result_future3 = server->AsyncInfer(*request3);

    // Get the infer result and check the result.
    auto result3 = result_future3.get();
    if (result3->HasError()) {
      FAIL(result3->ErrorMsg());
    }
    
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Output the elapsed time
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;

    std::cout << "Ran inference on model '" << result1->ModelName()
              << "', version '" << result1->ModelVersion()
              << "', with request ID '" << result1->Id() << "'\n";
    std::cout << "Ran inference on model '" << result2->ModelName()
              << "', version '" << result2->ModelVersion()
              << "', with request ID '" << result2->Id() << "'\n";
    std::cout << "Ran inference on model '" << result3->ModelName()
              << "', version '" << result3->ModelVersion()
              << "', with request ID '" << result3->Id() << "'\n";


    std::shared_ptr<tds::Tensor> result1_out0 = result1->Output("gpu_0/softmax_1");
    std::shared_ptr<tds::Tensor> result2_out0 = result2->Output("gpu_0/softmax_1");
    std::shared_ptr<tds::Tensor> result3_out0 = result3->Output("gpu_0/softmax_1");

    // Get full response.
    std::cout << result1->DebugString() << std::endl;
    std::cout << result2->DebugString() << std::endl;
    std::cout << result3->DebugString() << std::endl;

    // Get the server metrics.
    std::string model_stat = server->ModelStatistics("resnet50", 1);
    std::cout << "\n\n\n=========Model Statistics===========\n"
              << model_stat << "\n";
    
    // Unload 'add_sub' model as we don't need it anymore.
    server->UnloadModel("resnet50");
  }
  catch (const tds::TritonException& ex) {
    std::cerr << "Error: " << ex.what();
    exit(1);
  }

  return 0;
}
