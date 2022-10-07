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
#include <triton/backend/backend_common.h>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/triton/logging.hpp>
#include <rapids_triton/triton/model_instance.hpp>

namespace triton {
namespace backend {
namespace rapids {
namespace triton_api {
template <typename ModelInstanceState>
auto* instance_finalize(TRITONBACKEND_ModelInstance* instance)
{
  auto* result = static_cast<TRITONSERVER_Error*>(nullptr);
  try {
    auto* instance_state = get_instance_state<ModelInstanceState>(*instance);
    if (instance_state != nullptr) {
      instance_state->unload();

      log_info(__FILE__, __LINE__) << "TRITONBACKEND_ModelInstanceFinalize: delete instance state";

      delete instance_state;
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
