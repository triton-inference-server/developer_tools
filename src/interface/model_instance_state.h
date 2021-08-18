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

#include <impl/names.h>
#include <triton/backend/backend_model_instance.h>

#include <cstdint>

namespace triton {
namespace backend {
namespace NAMESPACE {
struct ModelInstanceState : public BackendModelInstance {
  static std::unique_ptr<ModelInstanceState> Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  ModelInstanceState(ModelState* model_state,
                     TRITONBACKEND_ModelInstance* triton_model_instance,
                     char const* name, std::int32_t device_id);
  auto& get_model() { return model_; }

 private:
  RapidsModel model_;
};

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
