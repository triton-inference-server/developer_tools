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
#include <triton/core/tritonbackend.h>

#include <rapids_triton/build_control.hpp>
#include <rapids_triton/memory/detail/resource.hpp>
#include <rapids_triton/triton/device.hpp>
#ifdef TRITON_ENABLE_GPU
#include <rapids_triton/memory/detail/gpu_only/resource.hpp>
#else
#include <rapids_triton/memory/detail/cpu_only/resource.hpp>
#endif

namespace triton {
namespace backend {
namespace rapids {

inline void setup_memory_resource(device_id_t device_id, TRITONBACKEND_MemoryManager* triton_manager = nullptr) {
  detail::setup_memory_resource<IS_GPU_BUILD>(device_id, triton_manager);
}

}  // namespace rapids
}  // namespace backend
}  // namespace triton
