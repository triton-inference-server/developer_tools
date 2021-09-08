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

#include <string>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/triton/backend.hpp>
#include <rapids_triton/triton/logging.hpp>

namespace triton { namespace backend { namespace rapids { namespace triton_api {
  inline auto* initialize(TRITONBACKEND_Backend* backend) {
    auto* result = static_cast<TRITONSERVER_Error*>(nullptr);
    try {
      auto name = get_backend_name(*backend);

      log_info(__FILE__, __LINE__) << "TRITONBACKEND_Initialize: " << name;

      if (!check_backend_version(*backend)) {
        throw TritonException{
            Error::Unsupported,
            "triton backend API version does not support this backend"};
      }
    } catch (TritonException& err) {
      result = err.error();
    }
    return result;
  }
}}}}
