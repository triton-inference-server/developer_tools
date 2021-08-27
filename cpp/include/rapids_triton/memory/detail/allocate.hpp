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

#include <cuda_runtime_api.h>

#include <rapids_triton/build_control.hpp>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/triton/logging.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace triton { namespace backend { namespace rapids { namespace detail {
template <typename T>
struct dev_deallocater {
  void operator()(T* d_ptr) {
    if constexpr (IS_GPU_BUILD) {
      cudaFree(reinterpret_cast<void*>(d_ptr));
    } else {
      log_error(
        __FILE__,
        __LINE__,
        "ERROR: device deallocation cannot be performed in non-GPU build!"
      );
    }
  }
};

/**
 * @brief Allocate given number of elements on GPU and return device pointer
 */
template <typename T>
[[nodiscard]] T*
dev_allocate(std::size_t count, cudaStream_t stream)
{
  if constexpr (!IS_GPU_BUILD) {
    throw TritonException(Error::Internal, "device allocation attempted in non-GPU build");
  }
  auto* ptr_d =
      static_cast<T*>(rmm::mr::get_current_device_resource()->allocate(
          sizeof(T) * count, stream));
  return ptr_d;
}

}}}}  // namespace triton::backend::rapids::detail
