// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once
#include <memory>
#include <string>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/triton/deployment.hpp>
#include <rapids_triton/triton/device.hpp>
#include <triton/backend/backend_model_instance.h>

namespace triton { namespace backend { namespace rapids {
  /** Get the name of a Triton model instance from the instance itself */
  inline auto
  get_model_instance_name(TRITONBACKEND_ModelInstance& instance)
  {
    auto cname = static_cast<char const*>(nullptr);
    triton_check(TRITONBACKEND_ModelInstanceName(&instance, &cname));
    return std::string(cname);
  }

  /** Get the device on which a Triton model instance is loaded
   *
   * If this instance is loaded on the host, 0 will be returned. Otherwise the
   * GPU device id will be returned.*/
  inline auto
  get_device_id(TRITONBACKEND_ModelInstance& instance)
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
  inline auto
  get_deployment_type(TRITONBACKEND_ModelInstance& instance)
  {
    auto kind = GPUDeployment;
    triton_check(TRITONBACKEND_ModelInstanceKind(&instance, &kind));
    return kind;
  }

  /** Return the Triton model from one of its instances
   */
  inline auto*
  get_model_from_instance(TRITONBACKEND_ModelInstance& instance)
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
  void
  set_instance_state(
      TRITONBACKEND_ModelInstance& instance,
      std::unique_ptr<ModelInstanceStateType>&& model_instance_state)
  {
    triton_check(TRITONBACKEND_ModelInstanceSetState(
        &instance, reinterpret_cast<void*>(model_instance_state.release())));
  }

  /** Get model instance state from instance */
  template <typename ModelInstanceStateType>
  auto*
  get_instance_state(TRITONBACKEND_ModelInstance& instance)
  {
    auto instance_state = static_cast<ModelInstanceStateType*>(nullptr);
    triton_check(TRITONBACKEND_ModelInstanceState(
        &instance, reinterpret_cast<void**>(&instance_state)));
    return instance_state;
  }

}}}  // namespace triton::backend::rapids
