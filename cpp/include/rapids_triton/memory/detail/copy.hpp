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
#include <cstring>

#include <cuda_runtime_api.h>

#include <rapids_triton/memory/types.hpp>
#include <raft/cudart_utils.h>


namespace triton { namespace backend { namespace rapids { namespace detail {

/**
 * @brief Copy given number of elements from one place to another, with either
 * source or destination on device
 */
template <typename T>
void
dev_copy(T* dst, T const* src, std::size_t len, cudaStream_t stream)
{
  if constexpr (IS_GPU_BUILD) {
    try {
      raft::copy(dst, src, len, stream);
    } catch (const raft::cuda_error& err) {
      throw TritonException(Error::Internal, err.what());
    }
  } else {
    throw TritonException(
      Error::Internal,
      "copy to or from device memory cannot be used in CPU-only builds"
    );
  }
}

/**
 * @brief Copy given number of elements from one place to another, with either
 * source or destination on device
 */
template <typename T>
void
host_copy(T* dst, T const* src, std::size_t len)
{
  std::memcpy(dst, src, len * sizeof(T));
}

template<typename T>
void
copy(T* dst, T const* src, std::size_t len, cudaStream_t stream, MemoryType
    dst_type, MemoryType src_type) {
  if (dst_type == DeviceMemory || src_type == DeviceMemory) {
    if constexpr (IS_GPU_BUILD) {
      dev_copy(dst, src, len, stream);
    } else {
      throw TritonException(
        Error::Internal,
        "DeviceMemory copy cannot be used in CPU-only builds"
      );
    }
  } else {
    host_copy(dst, src, len);
  }
}

}}}}  // namespace triton::backend::rapids::detail
