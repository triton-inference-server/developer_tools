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

#include <model.h>
#include <names.h>
#include <shared_state.h>
#include <stdint.h>
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>
#include <triton/backend/backend_model_instance.h>

#include <triton/developer_tools/triton/api/execute.hpp>
#include <triton/developer_tools/triton/api/initialize.hpp>
#include <triton/developer_tools/triton/api/instance_finalize.hpp>
#include <triton/developer_tools/triton/api/instance_initialize.hpp>
#include <triton/developer_tools/triton/api/model_finalize.hpp>
#include <triton/developer_tools/triton/api/model_initialize.hpp>
#include <triton/developer_tools/triton/model_instance_state.hpp>
#include <triton/developer_tools/triton/model_state.hpp>

namespace triton {
namespace developer_tools {
namespace NAMESPACE {

using ModelState = backend::TritonModelState<ToolsSharedState>;
using ModelInstanceState =
    backend::ModelInstanceState<ToolsModel, ToolsSharedState>;

extern "C" {

/** Confirm that backend is compatible with Triton's backend API version
 */
TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
  return backend::triton_api::initialize(backend);
}

TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  return backend::triton_api::model_initialize<ModelState>(model);
}

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  return backend::triton_api::model_finalize<ModelState>(model);
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance) {
  return backend::triton_api::instance_initialize<ModelState,
                                                 ModelInstanceState>(instance);
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance) {
  return backend::triton_api::instance_finalize<ModelInstanceState>(instance);
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** raw_requests,
    uint32_t const request_count) {
  return backend::triton_api::execute<ModelState, ModelInstanceState>(
      instance, raw_requests, static_cast<std::size_t>(request_count));
}

}  // extern "C"

}  // namespace NAMESPACE
}  // namespace developer_tools
}  // namespace triton
