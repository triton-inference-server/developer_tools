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
#include <interface/model_state.h>
#include <triton/backend/backend_model_instance.h>

#include <cstdint>
#include <rapids_triton/triton/model_instance.hpp>

namespace triton {
namespace backend {
namespace NAMESPACE {
struct ModelInstanceState : public BackendModelInstance {
  ModelInstanceState(ModelState& model_state,
                     TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(&model_state, triton_model_instance),
        model_(model_state.get_shared_state(),
               rapids::get_device_id(*triton_model_instance), CudaStream(),
               Kind(),
               JoinPath({RepositoryPath(), std::to_string(Version()),
                         ArtifactFilename()})) {}

  auto& get_model() const { return model_; }

  void load() { model_->load(); }
  void unload() { model_->unload(); }

 private:
  RapidsModel model_;
};

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
