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
#include <triton/developer_tools/memory/detail/gpu_only/owned_device_buffer.hpp>
#else
#include <triton/developer_tools/memory/detail/cpu_only/owned_device_buffer.hpp>
#endif
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <triton/developer_tools/build_control.hpp>
#include <triton/developer_tools/exceptions.hpp>
#include <vector>

namespace triton {
namespace developer_tools {
namespace backend {
TEST(BackendTools, owned_device_buffer)
{
  auto data = std::vector<int>{1, 2, 3};
#ifdef TRITON_ENABLE_GPU
  auto device_id = 0;
  cudaGetDevice(&device_id);
  auto stream = cudaStream_t{};
  cudaStreamCreate(&stream);

  auto buffer   = detail::owned_device_buffer<int, IS_GPU_BUILD>(device_id, data.size(), stream);
  auto data_out = std::vector<int>(data.size());
  cudaMemcpy(static_cast<void*>(buffer.get()),
             static_cast<void*>(data.data()),
             sizeof(int) * data.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(static_cast<void*>(data_out.data()),
             static_cast<void*>(buffer.get()),
             sizeof(int) * data.size(),
             cudaMemcpyDeviceToHost);
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
  cudaStreamDestroy(stream);
#else
  // Workaround for ungraceful handling of multiple template parameters in
  // EXPECT_THROW
  using dev_buffer = detail::owned_device_buffer<int, IS_GPU_BUILD>;
  EXPECT_THROW(dev_buffer(0, data.size(), 0), TritonException);
#endif
}

}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
