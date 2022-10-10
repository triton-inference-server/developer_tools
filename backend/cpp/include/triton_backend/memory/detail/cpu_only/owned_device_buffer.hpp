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

#pragma once
#include <cstddef>
#include <type_traits>
#include <triton_backend/cpu_only/cuda_runtime_replacement.hpp>
#include <triton_backend/exceptions.hpp>
#include <triton_backend/memory/detail/owned_device_buffer.hpp>
#include <triton_backend/triton/device.hpp>

namespace triton {
namespace backend {
namespace dev_tools {
namespace detail {

template<typename T>
struct owned_device_buffer<T, false> {
  using non_const_T = std::remove_const_t<T>;
  owned_device_buffer(device_id_t device_id, std::size_t size, cudaStream_t stream)
  {
    throw TritonException(Error::Internal,
                          "Attempted to use device buffer in non-GPU build");
  }

  auto* get() const { return static_cast<T*>(nullptr); }
};

}  // namespace detail
}  // namespace dev_tools
}  // namespace backend
}  // namespace triton
