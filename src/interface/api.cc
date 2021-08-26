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
#include <stdint.h>
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
auto* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
  auto* result = static_cast<TRITONSERVER_Error*>(nullptr);
  try {
    auto name = rapids::get_backend_name(*backend);

    // TODO (wphicks)
    rapids::log_info(
        __FILE__, __LINE__,
        (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

    if (!rapids::check_backend_version(*backend)) {
      throw rapids::TritonException{
          rapids::Error::Unsupported,
          "triton backend API version does not support this backend"};
    }
  } catch (rapids::TritonException& err) {
    result = err.error();
  }
  return result;
}

auto* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  auto* result = static_cast<TRITONSERVER_Error*>(nullptr);
  try {
    auto name = rapids::get_model_name(*model);

    auto version = rapids::get_model_version(*model);

    // TODO (wphicks)
    rapids::log_info(__FILE__, __LINE__,
                     (std::string("TRITONBACKEND_ModelInitialize: ") + name +
                      " (version " + std::to_string(version) + ")")
                         .c_str());

    auto rapids_model_state = std::make_unique<ModelState>(*model);
    rapids_model_state->load();

    rapids::set_model_state(*model, std::move(rapids_model_state));
  } catch (rapids::TritonException& err) {
    result = err.error();
  }

  return result;
}

auto* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  auto* result = static_cast<TRITONSERVER_Error*>(nullptr);
  try {
    auto model_state = rapids::get_model_state<ModelState>(*model);
    if (model_state != nullptr) {
      model_state->get_shared_state()->unload();
    }

    rapids::log_info(__FILE__, __LINE__,
                     "TRITONBACKEND_ModelFinalize: delete model state");

    delete model_state;
  } catch (rapids::TritonException& err) {
    result = err.error();
  }

  return result;
}

auto* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance) {
  auto* result = static_cast<TRITONSERVER_Error*>(nullptr);
  try {
    auto name = rapids::get_model_instance_name(*instance);
    auto device_id = rapids::get_device_id(*instance);
    auto deployment_type = rapids::get_deployment_type(*instance);

    // TODO (wphicks)
    // TODO (wphicks): Replace InstanceGroupKindString call
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

auto* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance) {
  auto* result = static_cast<TRITONSERVER_Error*>(nullptr);
  try {
    auto* instance_state =
        rapids::get_instance_state<ModelInstanceState>(*instance);
    if (instance_state != nullptr) {
      instance_state->unload();

      rapids::log_info(
          __FILE__, __LINE__,
          "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

      delete instance_state;
    }
  } catch (rapids::TritonException& err) {
    result = err.error();
  }

  return result;
}

auto* TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance,
                                         TRITONBACKEND_Request** raw_requests,
                                         uint32_t const request_count) {
  auto* result = static_cast<TRITONSERVER_Error*>(nullptr);

  try {
    auto* model_state = rapids::get_model_state<ModelState>(*instance);
    auto* instance_state =
        rapids::get_instance_state<ModelInstanceState>(*instance);
    auto& model = instance_state->get_model();
    auto max_batch_size = model.get_config_param("max_batch_size");
    auto batch = Batch{raw_requests,
                       request_count,
                       model_state->TritonMemoryManager(),
                       model_state->EnablePinnedInput(),
                       model_state->EnablePinnedOutput(),
                       max_batch_size,
                       model.get_stream()};

    model.predict(batch);
    batch.finalize();
  } catch (rapids::TritonException& err) {
    result = err.error();
  }

  return result;
}

}  // extern "C"

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
