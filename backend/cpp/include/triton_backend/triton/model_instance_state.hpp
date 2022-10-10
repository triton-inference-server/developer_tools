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
#include <cstdint>
#include <memory>
#include <triton_backend/triton/model_instance.hpp>
#include <triton_backend/triton/model_state.hpp>

namespace triton {
namespace backend {
namespace dev_tools {

template <typename RapidsModel, typename RapidsSharedState>
struct ModelInstanceState : public BackendModelInstance {
  ModelInstanceState(TritonModelState<RapidsSharedState>& model_state,
                     TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(&model_state, triton_model_instance),
      model_(model_state.get_shared_state(),
             dev_tools::get_device_id(*triton_model_instance),
             CudaStream(),
             Kind(),
             JoinPath({model_state.RepositoryPath(),
                       std::to_string(model_state.Version()),
                       ArtifactFilename()}))
  {
  }

  auto& get_model() const { return model_; }

  void load() { model_.load(); }
  void unload() { model_.unload(); }

 private:
  RapidsModel model_;
};

}  // namespace dev_tools
}  // namespace backend
}  // namespace triton
