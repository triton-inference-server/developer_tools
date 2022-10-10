/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#else
#include <triton_backend/cpu_only/cuda_runtime_replacement.hpp>
#endif
#include <gtest/gtest.h>

#include <triton_backend/exceptions.hpp>
#include <string>

namespace triton {
namespace backend {
namespace dev_tools {

TEST(DevToolsTriton, default_except)
{
  try {
    throw TritonException();
  } catch (TritonException const& err) {
    EXPECT_EQ(std::string(err.what()), std::string("encountered unknown error"));
  }
}

TEST(DevToolsTriton, msg_except)
{
  auto msg = std::string("TEST ERROR MESSAGE");
  try {
    throw TritonException(Error::Internal, msg);
  } catch (TritonException const& err) {
    EXPECT_EQ(std::string(err.what()), msg);
  }
  try {
    throw TritonException(Error::Internal, msg.c_str());
  } catch (TritonException const& err) {
    EXPECT_EQ(std::string(err.what()), msg);
  }
  try {
    throw TritonException(Error::Internal, msg);
  } catch (TritonException const& err) {
    try {
      throw(TritonException(err.error()));
    } catch (TritonException const& err2) {
      EXPECT_EQ(std::string(err2.what()), msg);
    }
  }
}

TEST(DevToolsTriton, triton_check)
{
  auto msg = std::string("TEST ERROR MESSAGE");
  EXPECT_THROW(triton_check(TRITONSERVER_ErrorNew(Error::Internal, msg.c_str())), TritonException);
  triton_check(nullptr);
}

TEST(DevToolsTriton, cuda_check)
{
#ifdef TRITON_ENABLE_GPU
  EXPECT_THROW(cuda_check(cudaError::cudaErrorMissingConfiguration), TritonException);
  cuda_check(cudaError::cudaSuccess);
#else
  EXPECT_THROW(cuda_check(cudaError::cudaErrorNonGpuBuild), TritonException);
#endif
}

}  // namespace dev_tools
}  // namespace backend
}  // namespace triton
