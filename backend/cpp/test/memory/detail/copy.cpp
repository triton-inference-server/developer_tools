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
#include <triton_backend/memory/detail/gpu_only/copy.hpp>
#else
#include <triton_backend/memory/detail/cpu_only/copy.hpp>
#endif
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <triton_backend/build_control.hpp>
#include <triton_backend/memory/types.hpp>
#include <vector>

namespace triton {
namespace backend {
namespace rapids {
TEST(RapidsTriton, copy)
{
  auto data     = std::vector<int>{1, 2, 3};
  auto data_out = std::vector<int>(data.size());
  detail::copy(data_out.data(), data.data(), data.size(), 0, HostMemory, HostMemory);
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));

  data_out = std::vector<int>(data.size());
#ifdef TRITON_ENABLE_GPU
  auto* ptr_d = static_cast<int*>(nullptr);
  cudaMalloc(reinterpret_cast<void**>(&ptr_d), sizeof(int) * data.size());
  detail::copy(ptr_d, data.data(), data.size(), 0, DeviceMemory, HostMemory);

  cudaMemcpy(static_cast<void*>(data_out.data()),
             static_cast<void*>(ptr_d),
             sizeof(int) * data.size(),
             cudaMemcpyDeviceToHost);
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
  cudaFree(reinterpret_cast<void*>(ptr_d));
#else
  EXPECT_THROW(detail::copy(data_out.data(), data.data(), data.size(), 0, HostMemory, DeviceMemory),
               TritonException);
  EXPECT_THROW(detail::copy(data_out.data(), data.data(), data.size(), 0, DeviceMemory, HostMemory),
               TritonException);
#endif
}

}  // namespace rapids
}  // namespace backend
}  // namespace triton
