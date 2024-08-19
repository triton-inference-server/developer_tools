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
#endif
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>
#include <triton/developer_tools/build_control.hpp>
#include <triton/developer_tools/exceptions.hpp>
#include <triton/developer_tools/memory/buffer.hpp>
#include <triton/developer_tools/memory/types.hpp>
#include <vector>

namespace triton {
namespace developer_tools {
namespace backend {
TEST(BackendTools, default_buffer)
{
  auto buffer = Buffer<int>();
  EXPECT_EQ(buffer.mem_type(), HostMemory);
  EXPECT_EQ(buffer.size(), 0);
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(buffer.device(), 0);
  EXPECT_EQ(buffer.stream(), cudaStream_t{});
#ifdef TRITON_ENABLE_GPU
  auto stream = cudaStream_t{};
  cudaStreamCreate(&stream);
  buffer.set_stream(stream);
  EXPECT_EQ(buffer.stream(), stream);
  cudaStreamDestroy(stream);
#endif
}

TEST(BackendTools, device_buffer)
{
  auto data = std::vector<int>{1, 2, 3};
#ifdef TRITON_ENABLE_GPU
  auto buffer = Buffer<int>(data.size(), DeviceMemory, 0, 0);

  ASSERT_EQ(buffer.mem_type(), DeviceMemory);
  ASSERT_EQ(buffer.size(), data.size());
  ASSERT_NE(buffer.data(), nullptr);

  auto data_out = std::vector<int>(data.size());
  cudaMemcpy(static_cast<void*>(buffer.data()),
             static_cast<void*>(data.data()),
             sizeof(int) * data.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(static_cast<void*>(data_out.data()),
             static_cast<void*>(buffer.data()),
             sizeof(int) * data.size(),
             cudaMemcpyDeviceToHost);
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));

#else
  EXPECT_THROW(Buffer<int>(data.size(), DeviceMemory, 0, 0), TritonException);
#endif
}

TEST(BackendTools, non_owning_device_buffer)
{
  auto data = std::vector<int>{1, 2, 3};
#ifdef TRITON_ENABLE_GPU
  auto* ptr_d = static_cast<int*>(nullptr);
  cudaMalloc(reinterpret_cast<void**>(&ptr_d), sizeof(int) * data.size());
  cudaMemcpy(static_cast<void*>(ptr_d),
             static_cast<void*>(data.data()),
             sizeof(int) * data.size(),
             cudaMemcpyHostToDevice);
  auto buffer = Buffer<int>(ptr_d, data.size(), DeviceMemory);

  ASSERT_EQ(buffer.mem_type(), DeviceMemory);
  ASSERT_EQ(buffer.size(), data.size());
  ASSERT_EQ(buffer.data(), ptr_d);

  auto data_out = std::vector<int>(data.size());
  cudaMemcpy(static_cast<void*>(data_out.data()),
             static_cast<void*>(buffer.data()),
             sizeof(int) * data.size(),
             cudaMemcpyDeviceToHost);
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));

  cudaFree(reinterpret_cast<void*>(ptr_d));
#else
  ASSERT_THROW(Buffer<int>(data.data(), data.size(), DeviceMemory), TritonException);
#endif
}

TEST(BackendTools, host_buffer)
{
  auto data   = std::vector<int>{1, 2, 3};
  auto buffer = Buffer<int>(data.size(), HostMemory, 0, 0);

  ASSERT_EQ(buffer.mem_type(), HostMemory);
  ASSERT_EQ(buffer.size(), data.size());
  ASSERT_NE(buffer.data(), nullptr);

  std::memcpy(
    static_cast<void*>(buffer.data()), static_cast<void*>(data.data()), data.size() * sizeof(int));

  auto data_out = std::vector<int>(buffer.data(), buffer.data() + buffer.size());
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
}

TEST(BackendTools, non_owning_host_buffer)
{
  auto data   = std::vector<int>{1, 2, 3};
  auto buffer = Buffer<int>(data.data(), data.size(), HostMemory);

  ASSERT_EQ(buffer.mem_type(), HostMemory);
  ASSERT_EQ(buffer.size(), data.size());
  ASSERT_EQ(buffer.data(), data.data());

  auto data_out = std::vector<int>(buffer.data(), buffer.data() + buffer.size());
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
}

TEST(BackendTools, copy_buffer)
{
  auto data        = std::vector<int>{1, 2, 3};
  auto orig_buffer = Buffer<int>(data.data(), data.size(), HostMemory);
  auto buffer      = Buffer<int>(orig_buffer);

  ASSERT_EQ(buffer.mem_type(), HostMemory);
  ASSERT_EQ(buffer.size(), data.size());
  ASSERT_NE(buffer.data(), orig_buffer.data());

  auto data_out = std::vector<int>(buffer.data(), buffer.data() + buffer.size());
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
}

TEST(BackendTools, move_buffer)
{
  auto data   = std::vector<int>{1, 2, 3};
  auto buffer = Buffer<int>(Buffer<int>(data.data(), data.size(), HostMemory));

  ASSERT_EQ(buffer.mem_type(), HostMemory);
  ASSERT_EQ(buffer.size(), data.size());
  ASSERT_EQ(buffer.data(), data.data());

  auto data_out = std::vector<int>(buffer.data(), buffer.data() + buffer.size());
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
}

TEST(BackendTools, move_assignment_buffer)
{
  auto data = std::vector<int>{1, 2, 3};

#ifdef TRITON_ENABLE_GPU
  auto buffer = Buffer<int>{data.data(), data.size() - 1, DeviceMemory};
#else
  auto buffer = Buffer<int>{data.data(), data.size() - 1, HostMemory};
#endif
  buffer      = Buffer<int>{data.size(), HostMemory};

  ASSERT_EQ(buffer.mem_type(), HostMemory);
  ASSERT_EQ(buffer.size(), data.size());
}

}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
