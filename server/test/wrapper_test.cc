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

namespace tsw = triton::server::wrapper;

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
    tsw::TritonServer::Create(
        tsw::ServerOptions({"/invalid_model_repository"}));
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
    options_.logging_ = tsw::LoggingOptions(
        tsw::VerboseLevel(0), false, false, false, tsw::LogFormat::LOG_DEFAULT,
        "");
  }

  tsw::ServerOptions options_;
};

TEST_F(TritonServerTest, StartNone)
{
  // Start server with default mode (NONE)
  try {
    auto server = tsw::TritonServer::Create(options_);
    std::set<std::string> loaded_models = server->LoadedModels();
    ASSERT_EQ(loaded_models.size(), 2);
    ASSERT_NE(loaded_models.find("add_sub"), loaded_models.end());
    ASSERT_NE(loaded_models.find("add_sub_str"), loaded_models.end());
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, NoneLoadUnload)
{
  // Start server with NONE mode which explicit model control is not allowed
  try {
    auto server = tsw::TritonServer::Create(options_);
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
    options_.model_control_mode_ =
        tsw::ModelControlMode::MODEL_CONTROL_EXPLICIT;
    auto server = tsw::TritonServer::Create(options_);
    std::set<std::string> loaded_models = server->LoadedModels();
    ASSERT_EQ(loaded_models.size(), 0);
    server->LoadModel("add_sub");
    loaded_models = server->LoadedModels();
    ASSERT_EQ(loaded_models.size(), 1);
    ASSERT_EQ(*loaded_models.begin(), "add_sub");
    server->UnloadModel("add_sub");
    loaded_models = server->LoadedModels();
    ASSERT_EQ(loaded_models.size(), 0);
  }
  catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, InferMinimal)
{
  try {
    auto server = tsw::TritonServer::Create(options_);

    std::vector<int32_t> input_data;
    while (input_data.size() < 16) {
      input_data.emplace_back(input_data.size());
    }
    auto request = tsw::InferRequest::Create(tsw::InferOptions("add_sub"));
    for (const auto& name : std::vector<std::string>{"INPUT0", "INPUT1"}) {
      request->AddInput(
          name, tsw::Tensor(
                    reinterpret_cast<char*>(input_data.data()),
                    input_data.size() * sizeof(int32_t), tsw::DataType::INT32,
                    {16}, tsw::MemoryType::CPU, 0));
    }
    std::future<std::unique_ptr<tsw::InferResult>> result_future =
        server->AsyncInfer(*request);
    auto result = result_future.get();

    // Check result metadata
    ASSERT_EQ(result->ModelName(), "add_sub");
    ASSERT_EQ(result->ModelVersion(), "1");
    ASSERT_EQ(result->Id(), "");

    // OUTPUT0 -> sum
    {
      std::string out_name("OUTPUT0");
      std::shared_ptr<tsw::Tensor> out = result->Output(out_name);
      ASSERT_EQ(out->shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out->data_type_, tsw::DataType::INT32);
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
      std::shared_ptr<tsw::Tensor> out = result->Output(out_name);
      ASSERT_EQ(out->shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out->data_type_, tsw::DataType::INT32);
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
    auto server = tsw::TritonServer::Create(options_);

    std::vector<int32_t> input_data;
    std::vector<std::string> input_data_str;
    while (input_data.size() < 16) {
      input_data.emplace_back(input_data.size());
      input_data_str.emplace_back(std::to_string(input_data.back()));
    }

    auto request = tsw::InferRequest::Create(tsw::InferOptions("add_sub_str"));
    for (const auto& name : std::vector<std::string>{"INPUT0", "INPUT1"}) {
      request->AddInput(
          name, input_data_str.begin(), input_data_str.end(),
          tsw::DataType::BYTES, {16}, tsw::MemoryType::CPU, 0);
    }

    std::future<std::unique_ptr<tsw::InferResult>> result_future =
        server->AsyncInfer(*request);
    auto result = result_future.get();

    // Check result metadata
    ASSERT_EQ(result->ModelName(), "add_sub_str");
    ASSERT_EQ(result->ModelVersion(), "1");
    ASSERT_EQ(result->Id(), "");

    std::vector<std::string> out_str;
    std::vector<int64_t> shape;
    tsw::DataType datatype;
    // OUTPUT0 -> sum
    {
      std::string out_name("OUTPUT0");
      std::shared_ptr<tsw::Tensor> out = result->Output(out_name);
      ASSERT_EQ(out->shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out->data_type_, tsw::DataType::BYTES);
      out_str = result->StringData(out_name);
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(out_str[i], std::to_string(2 * input_data[i]));
      }
    }

    // OUTPUT1 -> diff
    {
      std::string out_name("OUTPUT1");
      std::shared_ptr<tsw::Tensor> out = result->Output(out_name);
      ASSERT_EQ(out->shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out->data_type_, tsw::DataType::BYTES);
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

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
