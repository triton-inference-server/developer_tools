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

#include <cuda_runtime_api.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapids_triton/tensor/dtype.hpp>
#include <rapids_triton/tensor/tensor.hpp>
#include <vector>

namespace triton {
namespace backend {
namespace rapids {

TEST(RapidsTriton, default_tensor) {
  auto tensor = Tensor<int>();
  EXPECT_EQ(tensor.buffer().size(), 0);
  EXPECT_EQ(tensor.shape().size(), 0);
}

TEST(RapidsTriton, move_buffer_tensor) {
  auto shape = std::vector<std::size_t>{2, 2};
  auto data = std::vector<int>{1, 2, 3, 4};
  auto tensor =
      Tensor<int>(shape, Buffer<int>{data.data(), data.size(), HostMemory});

  EXPECT_EQ(data.data(), tensor.data());
  EXPECT_EQ(data.size(), tensor.size());
  EXPECT_THAT(tensor.shape(), ::testing::ElementsAreArray(shape));

  EXPECT_EQ(tensor.dtype(), DTypeInt32);
  EXPECT_EQ(tensor.mem_type(), HostMemory);
  EXPECT_EQ(tensor.stream(), cudaStream_t{});
  EXPECT_EQ(tensor.device(), 0);

  auto data_out =
      std::vector<int>(tensor.data(), tensor.data() + tensor.size());
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
}

TEST(RapidsTriton, multi_buffer_tensor) {
  auto shape = std::vector<std::size_t>{2, 2};
  auto data = std::vector<int>{1, 2, 3, 4};

  auto all_buffers = std::vector<Buffer<int>>{};
  all_buffers.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(all_buffers),
                 [](auto& elem) {
                   return Buffer<int>{&elem, 1, DeviceMemory};
                 });
  auto tensor = Tensor<int>(shape, all_buffers.begin(), all_buffers.end(),
                            DeviceMemory, 0, cudaStream_t{});

  auto data_out = std::vector<int>(data.size());
  cudaMemcpy(static_cast<void*>(data_out.data()),
             static_cast<void*>(tensor.data()), sizeof(int) * tensor.size(),
             cudaMemcpyDeviceToHost);
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
}

}  // namespace rapids
}  // namespace backend
}  // namespace triton
