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

#include <chrono>
#include <cstring>
#include <iostream>
#include <nvtx3/nvtx3.hpp>
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


void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v Enable verbose logging" << std::endl;

  exit(1);
}

template <typename T>
void
GenerateInputData(std::vector<char>* input0_data, const int element_count)
{
  input0_data->resize(element_count * sizeof(T));
  for (size_t i = 0; i < element_count; ++i) {
    ((T*)input0_data->data())[i] = 1.0 * (std::rand()%256);
  }
}
}  // namespace

int
main(int argc, char** argv)
{
  int verbose_level = 0;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "v:r:")) != -1) {
    switch (opt) {
      case 'v':
        verbose_level = 1;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }


  try {
    // Use 'ServerOptions' object to initialize TritonServer. Here we set model
    // control mode to 'EXPLICIT' so that we are able to load and unload models
    // after startup.
    tds::ServerOptions options({"./holoscan_models"});
    options.logging_.verbose_ =
        tds::LoggingOptions::VerboseLevel(verbose_level);
    options.model_control_mode_ = tds::ModelControlMode::EXPLICIT;
    // Enable tracing. The tracing output file 'trace_file' can be found after
    // this example is completed.
    options.trace_ = std::make_shared<tds::Trace>(
        "trace_file", tds::Trace::Level::TIMESTAMPS, 1, -1, 0);
    auto server = tds::TritonServer::Create(options);

    // Load three differen ONNX models.
    server->LoadModel("bmode_perspective");
    server->LoadModel("aortic_stenosis");
    server->LoadModel("plax_chamber");

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
    auto request1 = tds::InferRequest::Create(tds::InferOptions("bmode_perspective"));
    auto request2 = tds::InferRequest::Create(tds::InferOptions("aortic_stenosis"));
    auto request3 =
        tds::InferRequest::Create(tds::InferOptions("plax_chamber"));

    // Generate input data
    std::vector<char> input0_data;
    GenerateInputData<float>(&input0_data, 230400);
    size_t input0_size = input0_data.size();

    std::vector<char> input1_data;
    GenerateInputData<float>(&input1_data, 270000);
    size_t input1_size = input1_data.size();

    std::vector<char> input2_data;
    GenerateInputData<float>(&input2_data, 307200);
    size_t input2_size = input2_data.size();


    // Use the iterator of input vector to add input data to a request.
    request1->AddInput(
        "input_1", input0_data.begin(), input0_data.end(),
        tds::DataType::FP32, {1, 240, 320, 3}, tds::MemoryType::CPU, 0);

    request2->AddInput(
        "input_1", input1_data.begin(), input1_data.end(), tds::DataType::FP32,
        {1, 300, 300, 3}, tds::MemoryType::CPU, 0);

    request3->AddInput(
        "input_1", input2_data.begin(), input2_data.end(), tds::DataType::FP32,
        {1, 320, 320, 3}, tds::MemoryType::CPU, 0);

    // For this inference, we provide pre-allocated buffer for output. The infer
    // result will be stored in-place to the buffer.
    std::shared_ptr<void> allocated_output0(malloc(112), free);
    std::shared_ptr<void> allocated_output1(malloc(8), free);
    std::shared_ptr<void> allocated_output2(malloc(2457600), free);

    tds::Tensor alloc_output0(
        reinterpret_cast<char*>(allocated_output0.get()), 112,
        tds::MemoryType::CPU, 0);
    tds::Tensor alloc_output1(
        reinterpret_cast<char*>(allocated_output1.get()), 8,
        tds::MemoryType::CPU, 0);
    tds::Tensor alloc_output2(
        reinterpret_cast<char*>(allocated_output2.get()), 2457600,
        tds::MemoryType::CPU, 0);

    request1->AddRequestedOutput("output_1", alloc_output0);
    request2->AddRequestedOutput("output_1", alloc_output1);
    request3->AddRequestedOutput("lambda_4", alloc_output2);

    // Warm-Up
    for (int i = 0; i < 10000; i++) {
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
    }


    std::chrono::duration<double, std::milli> total_elapsed;

    for (int i = 0; i < 100; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      {
        nvtx3::scoped_range loop{"three inferences"};  // Range for iteration

        {
          nvtx3::scoped_range loop{"bmode_perspective inference"};  // Range for iteration
          // Call 'AsyncInfer' function to run inference.
          auto result_future1 = server->AsyncInfer(*request1);

          // Get the infer result and check the result.
          auto result1 = result_future1.get();
          if (result1->HasError()) {
            FAIL(result1->ErrorMsg());
          }
        }

        {
          nvtx3::scoped_range loop{
              "aortic_stenosis inference"};  // Range for iteration
          auto result_future2 = server->AsyncInfer(*request2);

          // Get the infer result and check the result.
          auto result2 = result_future2.get();
          if (result2->HasError()) {
            FAIL(result2->ErrorMsg());
          }
        }

        {
          nvtx3::scoped_range loop{
              "plax_chamber inference"};  // Range for iteration
          auto result_future3 = server->AsyncInfer(*request3);

          // Get the infer result and check the result.
          auto result3 = result_future3.get();
          if (result3->HasError()) {
            FAIL(result3->ErrorMsg());
          }
        }
      }

      auto end = std::chrono::high_resolution_clock::now();

      // Calculate the elapsed time in milliseconds
      total_elapsed += end - start;
      std::cout << "Running Avg Elapsed time: " << total_elapsed.count() / i
                << " ms" << std::endl;
    }

    // Output the elapsed time
    std::cout << " Avg Elapsed time: " << total_elapsed.count() / 100 << " ms"
              << std::endl;

    // Get the server metrics.
    std::string model_stat = server->ModelStatistics("bmode_perspective", 1);
    std::cout << "\n\n\n=========Model Statistics===========\n"
              << model_stat << "\n";
    model_stat = server->ModelStatistics("aortic_stenosis", 1);
    std::cout << "\n\n\n=========Model Statistics===========\n"
              << model_stat << "\n";
    model_stat = server->ModelStatistics("plax_chamber", 1);
    std::cout << "\n\n\n=========Model Statistics===========\n"
              << model_stat << "\n";


    // Unload model as we don't need it anymore.
    server->UnloadModel("bmode_perspective");
    server->UnloadModel("aortic_stenosis");
    server->UnloadModel("plax_chamber");
  }
  catch (const tds::TritonException& ex) {
    std::cerr << "Error: " << ex.what();
    exit(1);
  }

  return 0;
}
