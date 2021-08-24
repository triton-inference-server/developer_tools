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
#include <cstddef>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <rapids_triton/batch/batch.hpp>
#include <rapids_triton/tensor/tensor.hpp>
#include <rapids_triton/triton/deployment.hpp>
#include <rapids_triton/utils/narrow.hpp>

namespace triton { namespace backend { namespace rapids {
  /**
   * @brief Stores shared state for multiple instances of the same model
   */
  struct SharedModelState {
    virtual void load() {
    }
    virtual void unload() {
    }
  };

  template<typename SharedState=SharedModelState>
  struct Model {

    /* virtual void predict(Batch& batch) {
     *   Fetch tensors in order and feed them to predict overload
     * };
     */

    /**
     * @brief Return the preferred memory type in which to store data for this
     * batch or std::nullopt to accept whatever Triton returns
     *
     * The base implementation of this method will require data on-host if the
     * model itself is deployed on the host OR if this backend has not been
     * compiled with GPU support. Otherwise, models deployed on device will
     * receive memory on device. Overriding this method will allow derived
     * model classes to select a preferred memory location based on properties
     * of the batch or to simply return std::nullopt if device memory or host
     * memory will do equally well.
     */
    virtual std::optional<MemoryType> preferred_mem_type(Batch& batch) const {
      return (IS_GPU_BUILD && deployment_type_ == GPUDeployment) ? DeviceMemory : HostMemory;
    }

    /**
     * @brief Whether or not pinned memory should be used for I/O with this model
     */
    virtual bool enable_pinned() const {
      return false;
    }

    /**
     * @brief Get input tensor of a particular named input for an entire batch
     */
    template<typename T>
    auto get_input(Batch& batch, std::string const& name, std::optional<MemoryType>> const& mem_type, cudaStream_t stream) const {
      return batch.get_input<T>(name, mem_type, device_id_, stream);
    }
    template<typename T>
    auto get_input(Batch& batch, std::string const& name, std::optional<MemoryType> const& mem_type) const {
      return get_input<T>(name, mem_type, device_id_, default_stream_);
    }
    template<typename T>
    auto get_input(Batch& batch, std::string const& name) const {
      return get_input<T>(name, preferred_mem_type(batch), device_id_, default_stream_);
    }

    private:
      std::shared_ptr<SharedState> shared_state_;
      device_id_t device_id_;
      cudaStream_t default_stream_;
      DeploymentType deployment_type_;
  };
}}}  // namespace triton::backend::rapids
