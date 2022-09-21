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
#include "gtest/gtest.h"

#include <exception>
#include "triton/core/tritonserver.h"
#include "triton/developer_tools/server_wrapper.h"

namespace tds = triton::developer_tools::server;

namespace {

TEST(TritonServer, LibraryVersionCheck)
{
  // Check that proper 'libtritonserver.so' is used
  uint32_t major = 0;
  uint32_t minor = 0;
  auto err = TRITONSERVER_ApiVersion(&major, &minor);
  ASSERT_TRUE(err == nullptr) << "Unexpected error from API version call";
  ASSERT_EQ(major, TRITONSERVER_API_VERSION_MAJOR) << "Mismatch major version";
  ASSERT_GE(minor, TRITONSERVER_API_VERSION_MINOR) << "Older minor version";
}

TEST(TritonServer, StartInvalidRepository)
{
  // Run server with invalid model repository
  try {
    tds::TritonServer::Create(
        tds::ServerOptions({"/invalid_model_repository"}));
  }
  catch (std::exception& ex) {
    ASSERT_STREQ(
        ex.what(), "Internal-failed to stat file /invalid_model_repository\n");
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

class TritonServerTest : public ::testing::Test {
 protected:
  TritonServerTest() : options_({"./models"})
  {
    options_.logging_ = tds::LoggingOptions(
        tds::LoggingOptions::VerboseLevel(0), false, false, false,
        tds::LoggingOptions::LogFormat::DEFAULT, "");
  }

  tds::ServerOptions options_;
};

void
CPUAllocator(
    const char* tensor_name, size_t byte_size,
    tds::MemoryType preferred_memory_type, int64_t preferred_memory_type_id,
    void** buffer, tds::MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  std::cout << "Using custom allocation function" << std::endl;

  *actual_memory_type = tds::MemoryType::CPU;
  *actual_memory_type_id = preferred_memory_type_id;

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    std::cout << "allocated " << byte_size << " bytes for result tensor "
              << tensor_name << std::endl;
  } else {
    void* allocated_ptr = malloc(byte_size);
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

    default:
      std::cerr << "error: unexpected buffer allocated in CUDA managed memory"
                << std::endl;
      break;
  }
}

TEST_F(TritonServerTest, StartNone)
{
  // Start server with default mode (NONE)
  try {
    auto server = tds::TritonServer::Create(options_);
    std::set<std::string> loaded_models = server->LoadedModels();
    ASSERT_EQ(loaded_models.size(), 4);
    ASSERT_NE(loaded_models.find("add_sub"), loaded_models.end());
    ASSERT_NE(loaded_models.find("add_sub_str"), loaded_models.end());
    ASSERT_NE(loaded_models.find("failing_infer"), loaded_models.end());
    ASSERT_NE(loaded_models.find("square_int32"), loaded_models.end());
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, NoneLoadUnload)
{
  // Start server with NONE mode which explicit model control is not allowed
  try {
    auto server = tds::TritonServer::Create(options_);
    server->LoadModel("add_sub");
    server->UnloadModel("add_sub");
  }
  catch (std::exception& ex) {
    ASSERT_STREQ(
        ex.what(),
        "Error - LoadModel: Unavailable-explicit model load / unload is not "
        "allowed if polling is enabled\n");
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, Explicit)
{
  try {
    options_.model_control_mode_ = tds::ModelControlMode::EXPLICIT;

    std::set<std::string> startup_models;
    startup_models.insert("add_sub");
    options_.startup_models_ = startup_models;

    auto server = tds::TritonServer::Create(options_);
    std::set<std::string> loaded_models = server->LoadedModels();
    ASSERT_EQ(loaded_models.size(), 1);
    ASSERT_EQ(*loaded_models.begin(), "add_sub");
    server->UnloadModel("add_sub");
    loaded_models = server->LoadedModels();
    ASSERT_EQ(loaded_models.size(), 0);

    server->LoadModel("add_sub_str");
    loaded_models = server->LoadedModels();
    ASSERT_EQ(loaded_models.size(), 1);
    ASSERT_EQ(*loaded_models.begin(), "add_sub_str");
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, InferMinimal)
{
  try {
    auto server = tds::TritonServer::Create(options_);

    std::vector<int32_t> input_data;
    while (input_data.size() < 16) {
      input_data.emplace_back(input_data.size());
    }
    auto request = tds::InferRequest::Create(tds::InferOptions("add_sub"));
    for (const auto& name : std::vector<std::string>{"INPUT0", "INPUT1"}) {
      request->AddInput(
          name, tds::Tensor(
                    reinterpret_cast<char*>(input_data.data()),
                    input_data.size() * sizeof(int32_t), tds::DataType::INT32,
                    {16}, tds::MemoryType::CPU, 0));
    }
    std::future<std::unique_ptr<tds::InferResult>> result_future =
        server->AsyncInfer(*request);
    auto result = result_future.get();
    ASSERT_FALSE(result->HasError()) << result->ErrorMsg();

    // Check result metadata
    ASSERT_EQ(result->ModelName(), "add_sub");
    ASSERT_EQ(result->ModelVersion(), "1");
    ASSERT_EQ(result->Id(), "");

    // OUTPUT0 -> sum
    {
      std::string out_name("OUTPUT0");
      std::shared_ptr<tds::Tensor> out = result->Output(out_name);
      ASSERT_EQ(out->shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out->data_type_, tds::DataType::INT32);
      ASSERT_EQ(out->byte_size_, (input_data.size() * sizeof(int32_t)));
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(
            reinterpret_cast<const int32_t*>(out->buffer_)[i],
            (2 * input_data[i]));
      }
    }

    // OUTPUT1 -> diff
    {
      std::string out_name("OUTPUT1");
      std::shared_ptr<tds::Tensor> out = result->Output(out_name);
      ASSERT_EQ(out->shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out->data_type_, tds::DataType::INT32);
      ASSERT_EQ(out->byte_size_, (input_data.size() * sizeof(int32_t)));
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(reinterpret_cast<const int32_t*>(out->buffer_)[i], 0);
      }
    }
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, InferString)
{
  try {
    auto server = tds::TritonServer::Create(options_);

    std::vector<int32_t> input_data;
    std::vector<std::string> input_data_str;
    while (input_data.size() < 16) {
      input_data.emplace_back(input_data.size());
      input_data_str.emplace_back(std::to_string(input_data.back()));
    }

    auto request = tds::InferRequest::Create(tds::InferOptions("add_sub_str"));
    for (const auto& name : std::vector<std::string>{"INPUT0", "INPUT1"}) {
      request->AddInput(
          name, input_data_str.begin(), input_data_str.end(),
          tds::DataType::BYTES, {16}, tds::MemoryType::CPU, 0);
    }

    std::future<std::unique_ptr<tds::InferResult>> result_future =
        server->AsyncInfer(*request);
    auto result = result_future.get();
    ASSERT_FALSE(result->HasError()) << result->ErrorMsg();

    // Check result metadata
    ASSERT_EQ(result->ModelName(), "add_sub_str");
    ASSERT_EQ(result->ModelVersion(), "1");
    ASSERT_EQ(result->Id(), "");

    std::vector<std::string> out_str;
    std::vector<int64_t> shape;
    tds::DataType datatype;
    // OUTPUT0 -> sum
    {
      std::string out_name("OUTPUT0");
      std::shared_ptr<tds::Tensor> out = result->Output(out_name);
      ASSERT_EQ(out->shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out->data_type_, tds::DataType::BYTES);
      out_str = result->StringData(out_name);
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(out_str[i], std::to_string(2 * input_data[i]));
      }
    }

    // OUTPUT1 -> diff
    {
      std::string out_name("OUTPUT1");
      std::shared_ptr<tds::Tensor> out = result->Output(out_name);
      ASSERT_EQ(out->shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out->data_type_, tds::DataType::BYTES);
      out_str = result->StringData(out_name);
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(out_str[i], "0");
      }
    }
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, InferFailed)
{
  try {
    auto server = tds::TritonServer::Create(options_);

    std::vector<int32_t> input_data;
    while (input_data.size() < 16) {
      input_data.emplace_back(input_data.size());
    }
    auto request =
        tds::InferRequest::Create(tds::InferOptions("failing_infer"));
    request->AddInput(
        "INPUT", tds::Tensor(
                     reinterpret_cast<char*>(input_data.data()),
                     input_data.size() * sizeof(int32_t), tds::DataType::INT32,
                     {16}, tds::MemoryType::CPU, 0));
    std::future<std::unique_ptr<tds::InferResult>> result_future =
        server->AsyncInfer(*request);
    auto result = result_future.get();
    ASSERT_TRUE(result->HasError());
    ASSERT_STREQ(result->ErrorMsg().c_str(), "Internal-An Error Occurred\n");
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, InferCustomAllocator)
{
  try {
    auto server = tds::TritonServer::Create(options_);

    std::shared_ptr<tds::Allocator> allocator(
        new tds::Allocator(CPUAllocator, ResponseRelease));
    auto infer_options = tds::InferOptions("add_sub");
    infer_options.custom_allocator_ = allocator;
    auto request = tds::InferRequest::Create(infer_options);

    std::vector<int32_t> input_data;
    while (input_data.size() < 16) {
      input_data.emplace_back(input_data.size());
    }
    for (const auto& name : std::vector<std::string>{"INPUT0", "INPUT1"}) {
      request->AddInput(
          name, tds::Tensor(
                    reinterpret_cast<char*>(input_data.data()),
                    input_data.size() * sizeof(int32_t), tds::DataType::INT32,
                    {16}, tds::MemoryType::CPU, 0));
    }
    std::future<std::unique_ptr<tds::InferResult>> result_future =
        server->AsyncInfer(*request);
    auto result = result_future.get();
    ASSERT_FALSE(result->HasError()) << result->ErrorMsg();

    // Check result metadata
    ASSERT_EQ(result->ModelName(), "add_sub");
    ASSERT_EQ(result->ModelVersion(), "1");
    ASSERT_EQ(result->Id(), "");

    // OUTPUT0 -> sum
    {
      std::string out_name("OUTPUT0");
      std::shared_ptr<tds::Tensor> out = result->Output(out_name);
      ASSERT_EQ(out->shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out->data_type_, tds::DataType::INT32);
      ASSERT_EQ(out->byte_size_, (input_data.size() * sizeof(int32_t)));
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(
            reinterpret_cast<const int32_t*>(out->buffer_)[i],
            (2 * input_data[i]));
      }
    }

    // OUTPUT1 -> diff
    {
      std::string out_name("OUTPUT1");
      std::shared_ptr<tds::Tensor> out = result->Output(out_name);
      ASSERT_EQ(out->shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out->data_type_, tds::DataType::INT32);
      ASSERT_EQ(out->byte_size_, (input_data.size() * sizeof(int32_t)));
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(reinterpret_cast<const int32_t*>(out->buffer_)[i], 0);
      }
    }
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, InferPreAllocatedBuffer)
{
  try {
    auto server = tds::TritonServer::Create(options_);

    std::vector<int32_t> input_data;
    while (input_data.size() < 16) {
      input_data.emplace_back(input_data.size());
    }
    auto request = tds::InferRequest::Create(tds::InferOptions("add_sub"));
    for (const auto& name : std::vector<std::string>{"INPUT0", "INPUT1"}) {
      request->AddInput(
          name, tds::Tensor(
                    reinterpret_cast<char*>(input_data.data()),
                    input_data.size() * sizeof(int32_t), tds::DataType::INT32,
                    {16}, tds::MemoryType::CPU, 0));
    }

    // Provide pre-allocated buffer for 'OUTPUT0' and use default allocator for
    // 'OUTPUT1'
    void* buffer_output0 = malloc(64);
    tds::Tensor output0(
        reinterpret_cast<char*>(buffer_output0), 64, tds::MemoryType::CPU, 0);
    request->AddRequestedOutput("OUTPUT0", output0);
    request->AddRequestedOutput("OUTPUT1");

    std::future<std::unique_ptr<tds::InferResult>> result_future =
        server->AsyncInfer(*request);
    auto result = result_future.get();
    ASSERT_FALSE(result->HasError()) << result->ErrorMsg();

    // Check result metadata
    ASSERT_EQ(result->ModelName(), "add_sub");
    ASSERT_EQ(result->ModelVersion(), "1");
    ASSERT_EQ(result->Id(), "");

    // OUTPUT0 -> sum
    {
      std::string out_name("OUTPUT0");
      std::shared_ptr<tds::Tensor> out = result->Output(out_name);
      ASSERT_EQ(out->shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out->data_type_, tds::DataType::INT32);
      ASSERT_EQ(out->byte_size_, (input_data.size() * sizeof(int32_t)));
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(
            reinterpret_cast<const int32_t*>(buffer_output0)[i],
            (2 * input_data[i]));
      }
    }

    // OUTPUT1 -> diff
    {
      std::string out_name("OUTPUT1");
      std::shared_ptr<tds::Tensor> out = result->Output(out_name);
      ASSERT_EQ(out->shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out->data_type_, tds::DataType::INT32);
      ASSERT_EQ(out->byte_size_, (input_data.size() * sizeof(int32_t)));
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(reinterpret_cast<const int32_t*>(out->buffer_)[i], 0);
      }
    }

    free(buffer_output0);
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, InferDecoupledMultipleResponses)
{
  try {
    auto server = tds::TritonServer::Create(options_);

    std::vector<int32_t> input_data = {3};
    auto request = tds::InferRequest::Create(tds::InferOptions("square_int32"));
    request->AddInput(
        "IN", tds::Tensor(
                  reinterpret_cast<char*>(input_data.data()),
                  input_data.size() * sizeof(int32_t), tds::DataType::INT32,
                  {1}, tds::MemoryType::CPU, 0));
    std::future<std::unique_ptr<tds::InferResult>> result_future =
        server->AsyncInfer(*request);

    // Retrieve results from multiple responses.
    std::vector<std::unique_ptr<tds::InferResult>> results;
    results.push_back(result_future.get());
    size_t size = results.size();
    int count = 0;
    for (size_t i = 0; i < size; i++) {
      if (results[i]) {
        ASSERT_FALSE(results[i]->HasError()) << results[i]->ErrorMsg();
        auto next_future = results[i]->GetNextResult();
        if (next_future) {
          results.push_back(next_future->get());
          size++;
        }
        ASSERT_EQ(results[i]->ModelName(), "square_int32");
        ASSERT_EQ(results[i]->ModelVersion(), "1");
        ASSERT_EQ(results[i]->Id(), "");
        count++;
      }
    }
    ASSERT_EQ(count, 3);

    // OUTPUT1 -> 3
    {
      for (auto& result : results) {
        if (result) {
          std::string out_name("OUT");
          std::shared_ptr<tds::Tensor> out = result->Output(out_name);
          ASSERT_EQ(out->shape_, std::vector<int64_t>{1});
          ASSERT_EQ(out->data_type_, tds::DataType::INT32);
          ASSERT_EQ(out->byte_size_, (input_data.size() * sizeof(int32_t)));
          for (size_t i = 0; i < input_data.size(); ++i) {
            EXPECT_EQ(reinterpret_cast<const int32_t*>(out->buffer_)[i], 3);
          }
        }
      }
    }
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, InferDecoupledZeroResponse)
{
  try {
    auto server = tds::TritonServer::Create(options_);

    std::vector<int32_t> input_data = {0};
    auto request = tds::InferRequest::Create(tds::InferOptions("square_int32"));
    request->AddInput(
        "IN", tds::Tensor(
                  reinterpret_cast<char*>(input_data.data()),
                  input_data.size() * sizeof(int32_t), tds::DataType::INT32,
                  {1}, tds::MemoryType::CPU, 0));
    std::future<std::unique_ptr<tds::InferResult>> result_future =
        server->AsyncInfer(*request);
    std::vector<std::unique_ptr<tds::InferResult>> results;
    results.push_back(result_future.get());
    size_t size = results.size();
    int count = 0;
    for (size_t i = 0; i < size; i++) {
      if (results[i]) {
        ASSERT_FALSE(results[i]->HasError()) << results[i]->ErrorMsg();
        auto next_future = results[i]->GetNextResult();
        if (next_future) {
          results.push_back(next_future->get());
          size++;
        }
        ASSERT_EQ(results[i]->ModelName(), "square_int32");
        ASSERT_EQ(results[i]->ModelVersion(), "1");
        ASSERT_EQ(results[i]->Id(), "");
        count++;
      }
    }
    ASSERT_EQ(count, 0);

    {
      for (auto& result : results) {
        ASSERT_FALSE(result) << "Unexpected response.";
      }
    }
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
