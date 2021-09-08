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
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>

namespace triton { namespace backend { namespace rapids { namespace triton_api {
  template<typename ModelState>
  auto* model_finalize(TRITONBACKEND_Model* model) {
    auto* result = static_cast<TRITONSERVER_Error*>(nullptr);
    try {
      auto model_state = get_model_state<ModelState>(*model);
      if (model_state != nullptr) {
        model_state->get_shared_state()->unload();
      }

      log_info(__FILE__, __LINE__) << "TRITONBACKEND_ModelFinalize: delete model state";

      delete model_state;
    } catch (TritonException& err) {
      result = err.error();
    }

    return result;
  }
}}}}
