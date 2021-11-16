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
#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#endif

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapids_triton/build_control.hpp>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/resource.hpp>

namespace triton {
namespace backend {
namespace rapids {

TEST(RapidsTriton, set_memory_resource)
{
#ifdef TRITON_ENABLE_GPU
  auto device_id = int{};
  cuda_check(cudaGetDevice(&device_id));
  EXPECT_EQ(rmm::mr::get_current_device_resource()->is_equal(rmm::mr::cuda_memory_resource{}),
            true);
  setup_memory_resource(device_id);
  EXPECT_EQ(rmm::mr::get_current_device_resource()->is_equal(rmm::mr::cuda_memory_resource{}),
            false);
#else
  setup_memory_resource(0);
#endif
}

}  // namespace rapids
}  // namespace backend
}  // namespace triton
