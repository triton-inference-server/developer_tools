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
#include <triton/backend/backend_common.h>
#include <triton/core/tritonbackend.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <rapids_triton/exceptions.hpp>
#include <string>

namespace triton {
namespace backend {
namespace rapids {

inline auto get_model_version(TRITONBACKEND_Model& model)
{
  auto version = std::uint64_t{};
  triton_check(TRITONBACKEND_ModelVersion(&model, &version));
  return version;
}

inline auto get_model_name(TRITONBACKEND_Model& model)
{
  auto* cname = static_cast<char const*>(nullptr);
  triton_check(TRITONBACKEND_ModelName(&model, &cname));
  return std::string(cname);
}

inline auto get_model_config(TRITONBACKEND_Model& model)
{
  auto* config_message = static_cast<TRITONSERVER_Message*>(nullptr);
  triton_check(TRITONBACKEND_ModelConfig(&model, 1, &config_message));

  auto* buffer   = static_cast<char const*>(nullptr);
  auto byte_size = std::size_t{};
  triton_check(TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  auto model_config = std::make_unique<common::TritonJson::Value>();
  auto* err         = model_config->Parse(buffer, byte_size);
  auto* result      = TRITONSERVER_MessageDelete(config_message);
  if (err != nullptr) { throw(TritonException(err)); }
  if (result != nullptr) { throw(TritonException(result)); }
  return model_config;
}

/**
 * @brief Set model state (as used by Triton) to given object
 *
 * This function accepts a unique_ptr to an object derived from a Triton
 * BackendModel object and sets it as the stored state for a model in the
 * Triton server. Note that this object is not the same as a RAPIDS-Triton
 * "SharedModelState" object. The object that Triton expects must wrap this
 * SharedModelState and provide additional interface compatibility.
 */
template <typename ModelStateType>
void set_model_state(TRITONBACKEND_Model& model, std::unique_ptr<ModelStateType>&& model_state)
{
  triton_check(TRITONBACKEND_ModelSetState(&model, reinterpret_cast<void*>(model_state.release())));
}

/** Given a model, return its associated ModelState object */
template <typename ModelStateType>
auto* get_model_state(TRITONBACKEND_Model& model)
{
  auto vstate = static_cast<void*>(nullptr);
  triton_check(TRITONBACKEND_ModelState(&model, &vstate));

  auto* model_state = reinterpret_cast<ModelStateType*>(vstate);

  return model_state;
}

}  // namespace rapids
}  // namespace backend
}  // namespace triton
