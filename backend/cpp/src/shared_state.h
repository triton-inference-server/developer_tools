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

namespace triton {
namespace backend {
namespace NAMESPACE {

/* Triton allows multiple instances of a single model to be instantiated at the
 * same time (e.g. on different GPUs). All instances of a model share access to
 * an object which manages any state that can be shared across all instances.
 * Any logic necessary for managing such state should be implemented in a
 * struct named DevToolsSharedState, as shown here. Models may access this shared
 * state object via the `get_shared_state` method, which returns a shared
 * pointer to the DevToolsSharedState object.
 *
 * Not all backends require shared state, so leaving this implementation empty
 * is entirely valid */

struct DevToolsSharedState : dev_tools::SharedModelState {
  DevToolsSharedState(std::unique_ptr<common::TritonJson::Value>&& config)
    : dev_tools::SharedModelState{std::move(config)}
  {
  }
  void load() {}
  void unload() {}
};

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
