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

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#else
#include <triton_backend/cpu_only/cuda_runtime_replacement.hpp>
#endif
#include <names.h>
#include <shared_state.h>

#include <memory>
#include <optional>
#include <triton_backend/batch/batch.hpp>        // rapids::Batch
#include <triton_backend/memory/types.hpp>       // rapids::MemoryType
#include <triton_backend/model/model.hpp>        // rapids::Model
#include <triton_backend/tensor/tensor.hpp>      // rapids::copy
#include <triton_backend/triton/deployment.hpp>  // rapids::DeploymentType
#include <triton_backend/triton/device.hpp>      // rapids::device_id_t

namespace triton {
namespace backend {
namespace NAMESPACE {

/* Any logic necessary to perform inference with a model and manage its data
 * should be implemented in a struct named DevToolsModel, as shown here */

struct DevToolsModel : rapids::Model<DevToolsSharedState> {
  /***************************************************************************
   * BOILERPLATE                                                             *
   * *********************************************************************** *
   * The following constructor can be copied directly into any model
   * implementation.
   **************************************************************************/
  DevToolsModel(std::shared_ptr<DevToolsSharedState> shared_state,
              rapids::device_id_t device_id,
              cudaStream_t default_stream,
              rapids::DeploymentType deployment_type,
              std::string const& filepath)
    : rapids::Model<DevToolsSharedState>(
        shared_state, device_id, default_stream, deployment_type, filepath)
  {
  }

  /***************************************************************************
   * BASIC FEATURES                                                          *
   * *********************************************************************** *
   * The only method that *must* be implemented for a viable model is the
   * `predict` method, but the others presented here are often used for basic
   * model implementations. Filling out these methods should take care of most
   * use cases.
   **************************************************************************/

  /***************************************************************************
   * predict                                                                 *
   * *********************************************************************** *
   * This method performs the actual inference step on input data. Implementing
   * a predict function requires four steps:
   * 1. Call `get_input` on the provided `Batch` object for each of the input
   *    tensors named in the config file for this backend. This provides a
   *    `Tensor` object containing the input data.
   * 2. Call `get_output` on the provided `Batch` object for each of the output
   *    tensors named in the config file for this backend. This provides a
   *    `Tensor` object to which output values can be written.
   * 3. Perform inference based on the input Tensors and store the results in
   *    the output Tensors. `some_tensor.data()` can be used to retrieve a raw
   *    pointer to the underlying data.
   * 4. Call the `finalize` method on all output tensors.
   **************************************************************************/
  void predict(rapids::Batch& batch) const
  {
    // 1. Acquire a tensor representing the input named "input__0"
    auto input = get_input<float>(batch, "input__0");
    // 2. Acquire a tensor representing the output named "output__0"
    auto output = get_output<float>(batch, "output__0");

    // 3. Perform inference. In this example, we simply copy the data from the
    // input to the output tensor.
    rapids::copy(output, input);

    // 4. Call finalize on all output tensors. In this case, we have just one
    // output, so we call finalize on it.
    output.finalize();
  }

  /***************************************************************************
   * load / unload                                                           *
   * *********************************************************************** *
   * These methods can be used to perform one-time loading/unloading of
   * resources when a model is created. For example, data representing the
   * model may be loaded onto the GPU in the `load` method and unloaded in the
   * `unload` method. This data will then remain loaded while the server is
   * running.
   *
   * While these methods take no arguments, it is typical to read any necessary
   * input from the model configuration file by using the `get_config_param`
   * method. Any parameters defined in the "parameters" section of the config
   * can be accessed by name in this way. The maximum batch size can also be
   * retrieved using the name "max_batch_size".
   *
   * These methods need not be explicitly implemented if no loading/unloading
   * logic is required, but we show them here for illustrative purposes.
   **************************************************************************/
  void load() {}
  void unload() {}

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
   * properties of the batch.
   *
   * Valid MemoryType options to return are rapids::HostMemory and
   * rapids::DeviceMemory.
   **************************************************************************/
  std::optional<rapids::MemoryType> preferred_mem_type(rapids::Batch& batch) const
  {
    return std::nullopt;
  }
};

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
