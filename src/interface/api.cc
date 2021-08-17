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

#include <impl/names.h>
#include <triton/backend/backend_common.h>
#include <triton/backend/backend_model.h>
#include <triton/backend/backend_model_instance.h>
#include <rapids_triton/triton/backend.hpp>
#include <rapids_triton/triton/logging.hpp>

namespace triton {
namespace backend {
namespace NAMESPACE {

extern "C" {

/** Confirm that backend is compatible with Triton's backend API version
 */
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  try {
    auto name = rapids::get_backend_name(*backend);

    rapids::log_info(
        __FILE__, __LINE__,
        (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

    if (!rapids::check_backend_version(*backend)) {
      throw rapids::TritonException{
          rapids::Error::Unsupported,
          "triton backend API version does not support this backend"};
    }
  }
  catch (TritonException& err) {
    return err.error();
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  try {
    auto name = get_model_name(*model);

    auto version = get_model_version(*model);

    log_info(
        __FILE__, __LINE__,
        (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
         std::to_string(version) + ")")
            .c_str());

    set_model_state(*model, ModelState::Create(*model));
  }
  catch (TritonException& err) {
    return err.error();
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  try {
    auto model_state = get_model_state<ModelState>(*model);
    if (model_state != nullptr) {
      model_state->get_shared_state()->unload();
    }

    log_info(
        __FILE__, __LINE__, "TRITONBACKEND_ModelFinalize: delete model state");

    delete model_state;
  }
  catch (TritonException& err) {
    return err.error();
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  // TODO (wphicks): Replace
  try {
    std::string name = get_model_instance_name(*instance);
    int32_t device_id = get_device_id(*instance);
    TRITONSERVER_InstanceGroupKind kind = get_instance_kind(*instance);

    log_info(
        __FILE__, __LINE__,
        (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
         TRITONSERVER_InstanceGroupKindString(kind) + " device " +
         std::to_string(device_id) + ")")
            .c_str());

    ModelState* model_state = get_model_state<ModelState>(*instance);

    // WH
    set_instance_state<ModelInstanceState>(
        *instance, ModelInstanceState::Create(model_state, instance));
  }
  catch (TritonException& err) {
    return err.error();
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  try {
    void* vstate;
    triton_check(TRITONBACKEND_ModelInstanceState(instance, &vstate));
    ModelInstanceState* instance_state =
        reinterpret_cast<ModelInstanceState*>(vstate);

    if (instance_state != nullptr) {
      // WH
      instance_state->UnloadFILModel();

      log_info(
          __FILE__, __LINE__,
          "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

      delete instance_state;
    }
  }
  catch (TritonException& err) {
    return err.error();
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** raw_requests,
    const uint32_t request_count)
{
  // TODO (wphicks)
  // auto model = ...;
  // auto batch_input = ...;
  // auto batch_output = ...;
  // model.predict(batch_input, batch_output);
}

}  // extern "C"

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
