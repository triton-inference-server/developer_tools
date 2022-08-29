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
#include <iostream>
#include <string>
#include "server_wrapper.h"

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

void
CompareResult(
    const std::vector<std::string>& input0_data,
    const std::vector<std::string>& input1_data,
    const std::vector<std::string>& result0_data,
    const std::vector<std::string>& result1_data,
    const std::vector<int32_t>& expected_sum,
    const std::vector<int32_t>& expected_diff)
{
  for (size_t i = 0; i < 16; ++i) {
    std::cout << input0_data[i] << " + " << input0_data[i] << " = "
              << result0_data[i] << std::endl;
    std::cout << input0_data[i] << " - " << input1_data[i] << " = "
              << result1_data[i] << std::endl;

    if (expected_sum[i] != std::stoi(result0_data[i])) {
      std::cerr << "error: incorrect sum" << std::endl;
      exit(1);
    }
    if (expected_diff[i] != std::stoi(result1_data[i])) {
      std::cerr << "error: incorrect difference" << std::endl;
      exit(1);
    }
  }
}

void
Check(
    tsw::Tensor& output0, tsw::Tensor& output1,
    const std::vector<std::string>& input0_data,
    const std::vector<std::string>& input1_data,
    const std::string& output0_name, const std::string& output1_name,
    const std::vector<std::string>& result0_data,
    const std::vector<std::string>& result1_data,
    const std::vector<int32_t>& expected_sum,
    const std::vector<int32_t>& expected_diff)
{
  for (auto& output : {output0, output1}) {
    if ((output.name_ != output0_name) && (output.name_ != output1_name)) {
      FAIL("unexpected output '" + output.name_ + "'");
    }

    if ((output.shape_.size() != 1) || (output.shape_[0] != 16)) {
      std::cerr << "error: received incorrect shapes for " << output.name_
                << std::endl;
      exit(1);
    }

    if (output.data_type_ != tsw::Wrapper_DataType::BYTES) {
      FAIL(
          "unexpected datatype '" +
          std::string(WrapperDataTypeString(output.data_type_)) + "' for '" +
          output.name_ + "'");
    }

    if (output.memory_type_ != tsw::Wrapper_MemoryType::CPU) {
      FAIL(
          "unexpected memory type, expected to be allocated in CPU, got " +
          std::string(WrapperMemoryTypeString(output.memory_type_)) + ", id " +
          std::to_string(output.memory_type_id_) + " for " + output.name_);
    }
  }

  if (result0_data.size() != 16) {
    std::cerr << "error: received incorrect number of strings for OUTPUT0: "
              << result0_data.size() << std::endl;
  }
  if (result1_data.size() != 16) {
    std::cerr << "error: received incorrect number of strings for OUTPUT1: "
              << result1_data.size() << std::endl;
  }

  CompareResult(
      input0_data, input1_data, result0_data, result1_data, expected_sum,
      expected_diff);
}

}  // namespace

int
main(int argc, char** argv)
{
  int verbose_level = 0;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:H:")) != -1) {
    switch (opt) {
      case 'v':
        verbose_level = 1;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  // Use 'ServerOptions' object to initialize TritonServer.
  tsw::ServerOptions options({"./models"});
  options.logging_.verbose_ = verbose_level;
  auto server = tsw::TritonServer::Create(options);

  // We use a simple model that takes 2 input tensors of 16 strings
  // each and returns 2 output tensors of 16 strings each. The input
  // strings must represent integers. One output tensor is the
  // element-wise sum of the inputs and one output is the element-wise
  // difference.
  std::string model_name = "add_sub_str";

  // Use 'LoadedModels' function to check if the model we need is loaded.
  std::set<std::string> loaded_models;
  FAIL_IF_ERR(server->LoadedModels(&loaded_models), "getting loaded models");
  if (loaded_models.find(model_name) == loaded_models.end()) {
    FAIL("Model '" + model_name + "' is not found.");
  }

  // Initialize 'InferRequest' with the name of the model that we want to run an
  // inference on.
  auto request = tsw::InferRequest(tsw::InferOptions(model_name));

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones. The input tensors
  // are the string representation of these values.
  std::vector<std::string> input0_data(16);
  std::vector<std::string> input1_data(16);
  std::vector<int32_t> expected_sum(16);
  std::vector<int32_t> expected_diff(16);
  for (size_t i = 0; i < 16; ++i) {
    input0_data[i] = std::to_string(i);
    input1_data[i] = std::to_string(1);
    expected_sum[i] = i + 1;
    expected_diff[i] = i - 1;
  }

  std::vector<int64_t> shape{16};

  // Add two input tensors to the inference request.
  FAIL_IF_ERR(
      request.AddInput("INPUT0", input0_data.begin(), input0_data.end(), shape),
      "adding input0");
  FAIL_IF_ERR(
      request.AddInput("INPUT1", input1_data.begin(), input1_data.end(), shape),
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

    // Get the result data as a vector of string.
    std::vector<std::string> result0_data;
    std::vector<std::string> result1_data;
    FAIL_IF_ERR(
        result.StringData("OUTPUT0", &result0_data),
        "unable to get data for OUTPUT0");
    if (result0_data.size() != 16) {
      std::cerr << "error: received incorrect number of strings for OUTPUT0: "
                << result0_data.size() << std::endl;
    }
    FAIL_IF_ERR(
        result.StringData("OUTPUT1", &result1_data),
        "unable to get data for OUTPUT1");

    Check(
        output0, output1, input0_data, input1_data, "OUTPUT0", "OUTPUT1",
        result0_data, result1_data, expected_sum, expected_diff);

    // Get full response.
    std::string debug_str;
    FAIL_IF_ERR(result.DebugString(&debug_str), "getting debug string");
    std::cout << debug_str << std::endl;
  }

  return 0;
}
