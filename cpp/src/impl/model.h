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
#include <impl/shared_state.h>

#include <rapids_triton/batch/batch.hpp>  // rapids::Batch
#include <rapids_triton/memory/types.hpp>  // rapids::MemoryType
#include <rapids_triton/model/model.hpp>   // rapids::Model

namespace triton {
namespace backend {
namespace NAMESPACE {

struct RapidsModel : rapids::Model<RapidsSharedState> {
  void load() {}
  void unload() {}
  void predict(rapids::Batch& batch) {
    auto input = model.get_input<float>(batch, "input__0");
    auto output = model.get_output<float>(batch, "output__0");
    copy(output, input);
    // TODO(wphicks): Read config and make this interesting
  }

  /***************************************************************************
   * ADVANCED FEATURES                                                       *
   * *********************************************************************** *
   * None of the following methods are required to be implemented in order to
   * create a valid model, but they are presented here for those who require
   * the additional functionality they provide.
   **************************************************************************/

  /***************************************************************************
   * preferred_mem_type / preferred_mem_type_in / preferred_mem_type_out     *
   * *********************************************************************** *
   * If implemented, `preferred_mem_type` allows for control over when input
   * and output data are provided on the host versus on device. In the case
   * that a model prefers to receive its input on-host but return output
   * on-device (or vice versa), `preferred_mem_type_in` and
   * `preferred_mem_type_out` can be used for even more precise control.
   *
   * In this example, we simply return `std::nullopt` to indicate that the
   * model has no preference on its input/output data locations. Note that the
   * Batch being processed is taken as input to this function to facilitate
   * implementations that may switch their preferred memory location based on
   * e.g. the size of the batch.
   *
   * Valid MemoryType options to return are rapids::HostMemory and
   * rapids::DeviceMemory.
   **************************************************************************/
  std::optional<rapids::MemoryType> preferred_mem_type(
      rapids::Batch& batch) const {
    return std::nullopt;
  }
};

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
