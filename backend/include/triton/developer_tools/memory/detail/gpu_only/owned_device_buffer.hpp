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
#include <triton/developer_tools/memory/detail/owned_device_buffer.hpp>
#include <triton/developer_tools/triton/device.hpp>
#include <triton/developer_tools/utils/device_setter.hpp>
#include <rmm/device_buffer.hpp>

namespace triton {
namespace developer_tools {
namespace backend {
namespace detail {

template<typename T>
struct owned_device_buffer<T, true> {
  using non_const_T = std::remove_const_t<T>;
  owned_device_buffer(device_id_t device_id, std::size_t size, cudaStream_t stream)
    : data_{[&device_id, &size, &stream]() {
      auto device_context = device_setter{device_id};
      return rmm::device_buffer{size * sizeof(T), rmm::cuda_stream_view{stream}};
    }()}
  {
  }

  auto* get() const { return reinterpret_cast<T*>(data_.data()); }

 private:
  mutable rmm::device_buffer data_;
};

}  // namespace detail
}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
