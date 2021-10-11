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
#include <cuda_runtime_api.h>
#include <triton/backend/backend_common.h>

#include <rapids_triton/build_control.hpp>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/memory/resource.hpp>
#include <rapids_triton/triton/backend.hpp>
#include <rapids_triton/triton/device.hpp>
#include <rapids_triton/triton/logging.hpp>
#include <string>

namespace triton {
namespace backend {
namespace rapids {
namespace triton_api {
inline auto* initialize(TRITONBACKEND_Backend* backend)
{
  auto* result = static_cast<TRITONSERVER_Error*>(nullptr);
  try {
    auto name = get_backend_name(*backend);

    log_info(__FILE__, __LINE__) << "TRITONBACKEND_Initialize: " << name;

    if (!check_backend_version(*backend)) {
      throw TritonException{Error::Unsupported,
                            "triton backend API version does not support this backend"};
    }
    if constexpr (IS_GPU_BUILD) {
      auto device_count = int{};
      auto cuda_err     = cudaGetDeviceCount(&device_count);
      if (device_count > 0 && cuda_err == cudaSuccess) {
        auto device_id = int{};
        cuda_check(cudaGetDevice(&device_id));
        auto* triton_manager = static_cast<TRITONBACKEND_MemoryManager*>(nullptr);
        triton_check(TRITONBACKEND_BackendMemoryManager(backend, &triton_manager));

        setup_memory_resource(static_cast<device_id_t>(device_id), triton_manager);
      }
    }
  } catch (TritonException& err) {
    result = err.error();
  }
  return result;
}
}  // namespace triton_api
}  // namespace rapids
}  // namespace backend
}  // namespace triton
