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
#endif
#include <triton/developer_tools/build_control.hpp>
#include <triton/developer_tools/exceptions.hpp>

namespace triton {
namespace developer_tools {
namespace backend {

/** Struct for setting cuda device within a code block */
struct device_setter {
  device_setter(device_id_t device) : prev_device_{} {
    if constexpr(IS_GPU_BUILD) {
      cuda_check(cudaGetDevice(&prev_device_));
      cuda_check(cudaSetDevice(device));
    } else {
      throw TritonException(Error::Internal, "Device setter used in non-GPU build");
    }
  }

  ~device_setter() {
    if constexpr(IS_GPU_BUILD) {
      cudaSetDevice(prev_device_);
    }
  }
 private:
  device_id_t prev_device_;
};

}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
