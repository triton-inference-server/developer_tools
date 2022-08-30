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

#include "server_wrapper.h"
#include "triton/core/tritonserver.h"
#include <exception>

namespace tsw = triton::server::wrapper;

namespace {

TEST(TritonServer, SanityCheck)
{
  // Sanity check that proper 'libtritonserver->so' is used
  uint32_t major = 0, minor = 0;
  auto err = TRITONSERVER_ApiVersion(&major, &minor);
  ASSERT_TRUE(err == nullptr) << "Unexpected error from API version call";
  ASSERT_EQ(major, TRITONSERVER_API_VERSION_MAJOR) << "Mismatch major version";
  ASSERT_GE(minor, TRITONSERVER_API_VERSION_MINOR) << "Older minor version";
}

TEST(TritonServer, StartInvalidRepository)
{
  // Run server with invalid model repository
  try {
    tsw::TritonServer::Create(tsw::ServerOptions({"/invalid_model_repository"}));
  } catch (std::exception& ex) {
    ASSERT_STREQ(ex.what(), "Internal-failed to stat file /invalid_model_repository\n");
  } catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

class TritonServerTest : public ::testing::Test {
 protected:
  TritonServerTest()
    : options_({"./models"}) {
    options_.logging_ = tsw::LoggingOptions(false, false, false, false, tsw::Wrapper_LogFormat::LOG_DEFAULT, "");
  }

  tsw::ServerOptions options_;
};

TEST_F(TritonServerTest, StartNone)
{
  // Start server with default mode (NONE)
  try {
    auto server = tsw::TritonServer::Create(options_);
    std::set<std::string> loaded_models;
    ASSERT_TRUE(server->LoadedModels(&loaded_models).IsOk());
    ASSERT_EQ(loaded_models.size(), 2);
    ASSERT_NE(loaded_models.find("add_sub"), loaded_models.end());
    ASSERT_NE(loaded_models.find("add_sub_str"), loaded_models.end());
  } catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, NoneLoadUnload)
{
  // Start server with NONE mode which explicit model control is not allowed
  try {
    auto server = tsw::TritonServer::Create(options_);
    ASSERT_FALSE(server->LoadModel("add_sub").IsOk());
    ASSERT_FALSE(server->UnloadModel("add_sub").IsOk());
  } catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, Explicit)
{
  try {
    options_.model_control_mode_ = tsw::MODEL_CONTROL_EXPLICIT;
    auto server = tsw::TritonServer::Create(options_);
    std::set<std::string> loaded_models;
    ASSERT_TRUE(server->LoadedModels(&loaded_models).IsOk());
    ASSERT_EQ(loaded_models.size(), 0);
    ASSERT_TRUE(server->LoadModel("add_sub").IsOk());
    ASSERT_TRUE(server->LoadedModels(&loaded_models).IsOk());
    ASSERT_EQ(loaded_models.size(), 1);
    ASSERT_EQ(*loaded_models.begin(), "add_sub");
    ASSERT_TRUE(server->UnloadModel("add_sub").IsOk());
    ASSERT_TRUE(server->LoadedModels(&loaded_models).IsOk());
    ASSERT_EQ(loaded_models.size(), 0);
  } catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, InferMinimal)
{
  try {
    tsw::Error err;
    auto server = tsw::TritonServer::Create(options_);

    std::vector<int32_t> input_data;
    while (input_data.size() < 16) {
      input_data.emplace_back(input_data.size());
    }
    std::vector<tsw::Tensor> inputs;
    for (const auto& name : std::vector<std::string>{"INPUT0", "INPUT1"}) {
      inputs.emplace_back(tsw::Tensor(name, reinterpret_cast<char*>(input_data.data()),
      input_data.size() * sizeof(int32_t), tsw::Wrapper_DataType::INT32, {16},
      tsw::Wrapper_MemoryType::CPU, 0));
    }
    
    auto request = tsw::InferRequest(tsw::InferOptions("add_sub"));
    for (const auto& input : inputs) {
      ASSERT_TRUE((err = request.AddInput(input)).IsOk()) << err.Message();
    }
    std::future<tsw::InferResult> result_future;
    ASSERT_TRUE((err = server->AsyncInfer(&result_future, request)).IsOk()) << err.Message();
    auto result = result_future.get();
    ASSERT_FALSE(result.HasError()) << result.ErrorMsg();

    // Check result metadata
    ASSERT_EQ(result.ModelName() , "add_sub");
    ASSERT_EQ(result.ModelVersion(), "1");
    ASSERT_EQ(result.Id(), "");

    // OUTPUT0 -> sum
    {
      std::string out_name("OUTPUT0");
      tsw::Tensor out(out_name);
      ASSERT_TRUE((err = result.Output(&out)).IsOk()) << err.Message();
      ASSERT_EQ(out.shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out.data_type_, tsw::Wrapper_DataType::INT32);
      ASSERT_EQ(out.byte_size_, (input_data.size() * sizeof(int32_t)));
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(reinterpret_cast<const int32_t*>(out.buffer_)[i], (2 * input_data[i]));
      }
    }

    // OUTPUT1 -> diff
    {
      std::string out_name("OUTPUT1");
      tsw::Tensor out(out_name);
      ASSERT_TRUE((err = result.Output(&out)).IsOk()) << err.Message();
      ASSERT_EQ(out.shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out.data_type_, tsw::Wrapper_DataType::INT32);
      ASSERT_EQ(out.byte_size_, (input_data.size() * sizeof(int32_t)));
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(reinterpret_cast<const int32_t*>(out.buffer_)[i], 0);
      }
    }
  } catch (...) {
    ASSERT_NO_THROW(throw);
  }
}

TEST_F(TritonServerTest, InferString)
{
  try {
    tsw::Error err;
    auto server = tsw::TritonServer::Create(options_);

    std::vector<int32_t> input_data;
    std::vector<std::string> input_data_str;
    while (input_data.size() < 16) {
      input_data.emplace_back(input_data.size());
      input_data_str.emplace_back(std::to_string(input_data.back()));
    }

    auto request = tsw::InferRequest(tsw::InferOptions("add_sub_str"));
    for (const auto& name : std::vector<std::string>{"INPUT0", "INPUT1"}) {
      ASSERT_TRUE((err = request.AddInput(name, input_data_str.begin(), input_data_str.end(), {16})).IsOk()) << err.Message();
    }
    
    std::future<tsw::InferResult> result_future;
    ASSERT_TRUE((err = server->AsyncInfer(&result_future, request)).IsOk()) << err.Message();
    auto result = result_future.get();
    ASSERT_FALSE(result.HasError()) << result.ErrorMsg();

    // Check result metadata
    ASSERT_EQ(result.ModelName() , "add_sub_str");
    ASSERT_EQ(result.ModelVersion(), "1");
    ASSERT_EQ(result.Id(), "");

    std::vector<std::string> out_str;
    std::vector<int64_t> shape;
    tsw::Wrapper_DataType datatype;
    // OUTPUT0 -> sum
    {
      std::string out_name("OUTPUT0");
      tsw::Tensor out(out_name);
      ASSERT_TRUE((err = result.Output(&out)).IsOk()) << err.Message();
      ASSERT_EQ(out.shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out.data_type_, tsw::Wrapper_DataType::BYTES);
      ASSERT_TRUE((err = result.StringData(out_name, &out_str)).IsOk()) << err.Message();
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(out_str[i], std::to_string(2 * input_data[i]));
      }
    }

    // OUTPUT1 -> diff
    {
      std::string out_name("OUTPUT1");
      tsw::Tensor out(out_name);
      ASSERT_TRUE((err = result.Output(&out)).IsOk()) << err.Message();
      ASSERT_EQ(out.shape_, std::vector<int64_t>{16});
      ASSERT_EQ(out.data_type_, tsw::Wrapper_DataType::BYTES);
      ASSERT_TRUE((err = result.StringData(out_name, &out_str)).IsOk()) << err.Message();
      for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_EQ(out_str[i], "0");
      }
    }
  } catch (...) {
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
