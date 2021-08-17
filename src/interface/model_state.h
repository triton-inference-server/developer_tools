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
#include <triton/backend/backend_model.h>

namespace triton { namespace backend { namespace NAMESPACE {
struct ModelState : public BackendModel {
  static std::unique_ptr<ModelState> Create(TRITONBACKEND_Model& triton_model);

  ModelState(
      TRITONBACKEND_Model* triton_model, const char* name,
      const uint64_t version);

  auto get_shared_state() { return state_; }

 private:
  std::shared_ptr<RapidsSharedState> state_;
};

}}}  // namespace triton::backend::NAMESPACE
