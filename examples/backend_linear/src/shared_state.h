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

#include <names.h>

#include <memory>
#include <triton_backend/model/shared_state.hpp>
#include <triton_backend/triton/logging.hpp>
#include <vector>

namespace triton {
namespace backend {
namespace NAMESPACE {

struct RapidsSharedState : dev_tools::SharedModelState {
  RapidsSharedState(std::unique_ptr<common::TritonJson::Value>&& config)
      : dev_tools::SharedModelState{std::move(config)} {}
  void load() { alpha = get_config_param<float>("alpha"); }
  void unload() {
    dev_tools::log_info(__FILE__, __LINE__) << "Unloading shared state...";
  }

  float alpha = 1.0f;
};

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
