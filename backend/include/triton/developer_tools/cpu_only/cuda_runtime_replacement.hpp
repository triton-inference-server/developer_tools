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
#else

namespace triton {
namespace developer_tools {
namespace backend {

using cudaStream_t = void*;

enum struct cudaError_t {cudaSuccess, cudaErrorNonGpuBuild};
using cudaError = cudaError_t;
auto constexpr cudaSuccess = cudaError_t::cudaSuccess;

inline void cudaGetLastError() {}

inline auto const * cudaGetErrorString(cudaError_t err) {
  return "CUDA function used in non-GPU build";
}

inline auto cudaStreamSynchronize(cudaStream_t stream) {
  return cudaError_t::cudaErrorNonGpuBuild;
}

inline auto cudaGetDevice(int* device_id) {
  return cudaError_t::cudaErrorNonGpuBuild;
}

inline auto cudaGetDeviceCount(int* count) {
  return cudaError_t::cudaErrorNonGpuBuild;
}


}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
#endif
