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

namespace triton { namespace backend { namespace rapids { namespace triton_api {
  template<typename ModelState, typename ModelInstanceState>
  auto* instance_initialize(TRITONBACKEND_ModelInstance* instance) {
    auto* result = static_cast<TRITONSERVER_Error*>(nullptr);
    try {
      auto name = rapids::get_model_instance_name(*instance);
      auto device_id = rapids::get_device_id(*instance);
      auto deployment_type = rapids::get_deployment_type(*instance);

      // TODO (wphicks): Use sstream
      rapids::log_info(__FILE__, __LINE__,
                       (std::string("TRITONBACKEND_ModelInstanceInitialize: ") +
                        name + " (" + TRITONSERVER_InstanceGroupKindString(kind) +
                        " device " + std::to_string(device_id) + ")")
                           .c_str());

      auto* model_state = rapids::get_model_state<ModelState>(*instance);

      auto rapids_model =
          std::make_unique<ModelInstanceState>(model_state, instance)

              rapids::set_instance_state(*instance, std::move(rapids_model));
    } catch (rapids::TritonException& err) {
      result = err.error();
    }
    return result;
  }
}}}}
