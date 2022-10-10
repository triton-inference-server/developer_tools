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
#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#include <raft/cudart_utils.h>
#endif

#include <cstddef>
#include <cstring>
#include <triton_backend/memory/types.hpp>
#include <triton_backend/exceptions.hpp>

namespace triton {
namespace backend {
namespace rapids {
namespace detail {

template <typename T>
void copy(T* dst,
          T const* src,
          std::size_t len,
          cudaStream_t stream,
          MemoryType dst_type,
          MemoryType src_type)
{
  if (dst_type == DeviceMemory || src_type == DeviceMemory) {
    try {
      raft::copy(dst, src, len, stream);
    } catch (raft::cuda_error const& err) {
      throw TritonException(Error::Internal, err.what());
    }
  } else {
    std::memcpy(dst, src, len * sizeof(T));
  }
}

}  // namespace detail
}  // namespace rapids
}  // namespace backend
}  // namespace triton
