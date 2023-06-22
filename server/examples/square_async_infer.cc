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

#include <string>

#include "triton/developer_tools/server_wrapper.h"

namespace tds = triton::developer_tools::server;

namespace {

#define FAIL(MSG)                                 \
  do {                                            \
    std::cerr << "error: " << (MSG) << std::endl; \
    exit(1);                                      \
  } while (false)

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
GetResults(
    std::vector<std::unique_ptr<tds::InferResult>>& results,
    std::future<std::unique_ptr<tds::InferResult>> future)
{
  results.push_back(future.get());
  size_t size = results.size();
  for (size_t i = 0; i < size; i++) {
    if (results[i]) {
      if (results[i]->HasError()) {
        FAIL(results[i]->ErrorMsg());
      }
      auto next_future = results[i]->GetNextResult();
      if (next_future) {
        results.push_back(next_future->get());
        size++;
      }
    }
  }
}

void
Check(
    const std::vector<std::unique_ptr<tds::InferResult>>& results,
    int32_t input_value)
{
  int count = 0;
  std::cout << "Outputs:\n";
  for (auto& result : results) {
    if (result) {
      std::shared_ptr<tds::Tensor> out = result->Output("OUT");

      if ((out->shape_.size() != 1) || (out->shape_[0] != 1)) {
        FAIL("error: received incorrect shapes");
      }

      if (out->memory_type_ != tds::MemoryType::CPU) {
        FAIL(
            "unexpected memory type, expected to be allocated in CPU, got " +
            std::string(MemoryTypeString(out->memory_type_)) + ", id " +
            std::to_string(out->memory_type_id_));
      }

      if (out->data_type_ != tds::DataType::INT32) {
        FAIL(
            "unexpected datatype '" +
            std::string(DataTypeString(out->data_type_)));
      }

      if (input_value && (*((int32_t*)out->buffer_) != input_value)) {
        FAIL(
            "incorrect value, expected: '" + std::to_string(input_value) +
            ", got :" + std::to_string(*((int32_t*)out->buffer_)));
      }

      std::cout << *((int32_t*)out->buffer_) << "\n";
      count++;
    }
  }

  if (count != input_value) {
    std::cerr << "error: received incorrect number of responses. Expected: "
              << input_value << ", got: " << count << std::endl;
  }
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
  try {
    // Use 'ServerOptions' object to initialize TritonServer.
    tds::ServerOptions options({"./models"});
    options.logging_.verbose_ =
        tds::LoggingOptions::VerboseLevel(verbose_level);
    options.model_control_mode_ = tds::ModelControlMode::EXPLICIT;
    auto server = tds::TritonServer::Create(options);

    // Load 'square_int32' model.
    server->LoadModel("square_int32");

    // Please see here for more information about this decoupled model:
    // https://github.com/triton-inference-server/python_backend/tree/main/examples/decoupled.
    std::string model_name = "square_int32";

    // Initialize 'InferRequest' with the name of the model that we want to run
    // an inference on.
    auto request1 = tds::InferRequest::Create(tds::InferOptions(model_name));

    // Create the data for an input tensor. For square model, value '3' here
    // means there will be three reponses for this request, and each reponse
    // contains only one output with value '3'.
    std::vector<int32_t> input_data1 = {3};

    std::vector<int64_t> shape{1};

    // Add input tensor to the inference request.
    request1->AddInput(
        "IN", input_data1.begin(), input_data1.end(), tds::DataType::INT32,
        shape, tds::MemoryType::CPU, 0);

    // Call 'AsyncInfer' function to run inference.
    auto result_future1 = server->AsyncInfer(*request1);

    // Run the second inference.
    auto request2 = tds::InferRequest::Create(tds::InferOptions(model_name));

    // Create the data for an input tensor. For square model, value '0' here
    // means there won't be any reponses for this request.
    std::vector<int32_t> input_data2 = {0};
    request2->AddInput(
        "IN", input_data2.begin(), input_data2.end(), tds::DataType::INT32,
        shape, tds::MemoryType::CPU, 0);

    // Call 'AsyncInfer' function to run inference.
    auto result_future2 = server->AsyncInfer(*request2);

    // Get the infer results from both inferences and check the results.
    std::vector<std::unique_ptr<tds::InferResult>> results1;
    GetResults(results1, std::move(result_future1));
    Check(results1, 3);

    std::vector<std::unique_ptr<tds::InferResult>> results2;
    GetResults(results2, std::move(result_future2));
    Check(results2, 0);
  }
  catch (const tds::TritonException& ex) {
    std::cerr << "Error: " << ex.what();
    exit(1);
  }

  return 0;
}
