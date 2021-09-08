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

#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/triton/logging.hpp>
#include <rapids_triton/triton/model.hpp>
#include <rapids_triton/triton/model_instance.hpp>
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model_instance.h>

namespace triton { namespace backend { namespace rapids { namespace triton_api {
  template<typename ModelState, typename ModelInstanceState>
  auto* instance_initialize(TRITONBACKEND_ModelInstance* instance) {
    auto* result = static_cast<TRITONSERVER_Error*>(nullptr);
    try {
      auto name = get_model_instance_name(*instance);
      auto device_id = get_device_id(*instance);
      auto deployment_type = get_deployment_type(*instance);

      log_info(__FILE__, __LINE__)
        << "TRITONBACKEND_ModelInstanceInitialize: "
        << name
        << " ("
        << TRITONSERVER_InstanceGroupKindString(deployment_type)
        << " device "
        << device_id
        << ")";

      auto* triton_model = get_model_from_instance(*instance);
      auto* model_state = get_model_state<ModelState>(*triton_model);

      auto rapids_model =
          std::make_unique<ModelInstanceState>(*model_state, instance);

      set_instance_state<ModelInstanceState>(*instance, std::move(rapids_model));
    } catch (TritonException& err) {
      result = err.error();
    }
    return result;
  }
}}}}
