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

#include <cstddef>

#include <cuda_runtime_api.h>
#include <names.h>
#include <gpu_infer.h>
#include <triton_backend/batch/batch.hpp>
#include <triton_backend/tensor/tensor.hpp>

namespace triton { namespace backend { namespace NAMESPACE {

namespace {
__global__ void cu_gpu_infer(float* r, float const* u, float const* v,
                             float* c, float alpha, std::size_t features,
                             std::size_t length) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < length) {
    r[id] = alpha * u[id] + v[id] + c[id % features];
  }
}
}

void gpu_infer(float* r, float const* u, float const* v, float* c, float alpha,
               std::size_t features, std::size_t length, cudaStream_t stream) {
  auto constexpr block_size = 1024;
  auto grid_size = static_cast<int>(std::max(1.0f, std::ceil(length /
          static_cast<float>(block_size))));;
  cu_gpu_infer<<<grid_size, block_size, 0, stream>>>(r, u, v, c, alpha, features, length);
}

}}}
