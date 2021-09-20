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
#include <triton/backend/backend_model_instance.h>
#include <memory>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/triton/deployment.hpp>
#include <rapids_triton/triton/device.hpp>
#include <string>

namespace triton {
namespace backend {
namespace rapids {
/** Get the name of a Triton model instance from the instance itself */
inline auto get_model_instance_name(TRITONBACKEND_ModelInstance& instance)
{
  auto cname = static_cast<char const*>(nullptr);
  triton_check(TRITONBACKEND_ModelInstanceName(&instance, &cname));
  return std::string(cname);
}

/** Get the device on which a Triton model instance is loaded
 *
 * If this instance is loaded on the host, 0 will be returned. Otherwise the
 * GPU device id will be returned.*/
inline auto get_device_id(TRITONBACKEND_ModelInstance& instance)
{
  auto device_id = device_id_t{};
  triton_check(TRITONBACKEND_ModelInstanceDeviceId(&instance, &device_id));
  return device_id;
}

/** Determine how a Triton model instance is deployed
 *
 * Returns enum value indicating whether the instance is deployed on device
 * or on the host
 */
inline auto get_deployment_type(TRITONBACKEND_ModelInstance& instance)
{
  auto kind = GPUDeployment;
  triton_check(TRITONBACKEND_ModelInstanceKind(&instance, &kind));
  return kind;
}

/** Return the Triton model from one of its instances
 */
inline auto* get_model_from_instance(TRITONBACKEND_ModelInstance& instance)
{
  auto* model = static_cast<TRITONBACKEND_Model*>(nullptr);
  triton_check(TRITONBACKEND_ModelInstanceModel(&instance, &model));
  return model;
}

/**
 * @brief Set Triton model instance state to given object
 *
 * This function accepts a unique_ptr to an object derived from a Triton
 * BackendModelInstance object and sets it as the stored state for a model in the
 * Triton server. Note that this object is not the same as a RAPIDS-Triton
 * "Model" object. The object that Triton expects must wrap this Model and
 * provide additional interface compatibility.
 */
template <typename ModelInstanceStateType>
void set_instance_state(TRITONBACKEND_ModelInstance& instance,
                        std::unique_ptr<ModelInstanceStateType>&& model_instance_state)
{
  triton_check(TRITONBACKEND_ModelInstanceSetState(
    &instance, reinterpret_cast<void*>(model_instance_state.release())));
}

/** Get model instance state from instance */
template <typename ModelInstanceStateType>
auto* get_instance_state(TRITONBACKEND_ModelInstance& instance)
{
  auto instance_state = static_cast<ModelInstanceStateType*>(nullptr);
  triton_check(
    TRITONBACKEND_ModelInstanceState(&instance, reinterpret_cast<void**>(&instance_state)));
  return instance_state;
}

}  // namespace rapids
}  // namespace backend
}  // namespace triton
