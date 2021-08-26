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
#include <rapids_triton/model/shared_state.hpp>
#include <rapids_triton/tensor/tensor.hpp>
#include <rapids_triton/triton/deployment.hpp>
#include <rapids_triton/triton/device.hpp>
#include <rapids_triton/utils/narrow.hpp>

namespace triton { namespace backend { namespace rapids {
  template<typename SharedState=SharedModelState>
  struct Model {

    virtual void predict(Batch& batch) = 0;

    virtual void load() {}
    virtual void unload() {}

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
    virtual std::optional<MemoryType> preferred_mem_type_in(Batch& batch) const {
      return preferred_mem_type(batch);
    }
    virtual std::optional<MemoryType> preferred_mem_type_out(Batch& batch) const {
      return preferred_mem_type(batch);
    }

    /**
     * @brief Retrieve a stream used to set up batches for this model
     *
     * The base implementation of this method simply returns the default stream
     * provided by Triton for use with this model. Child classes may choose to
     * override this in order to provide different streams for use with
     * successive incoming batches. For instance, one might cycle through
     * several streams in order to distribute batches across them, but care
     * should be taken to ensure proper synchronization in this case. It is
     * recommended that this method be overridden only when strictly necessary.
     */
    virtual cudaStream_t get_stream() {
      return default_stream_;
    }

    /**
     * @brief Get input tensor of a particular named input for an entire batch
     */
    //TODO(wphicks): Handle const for input type
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

    /**
     * @brief Get output tensor of a particular named output for an entire batch
     */
    template<typename T>
    auto get_output(Batch& batch, std::string const& name, std::optional<MemoryType>> const& mem_type, cudaStream_t stream) const {
      return batch.get_output<T>(name, mem_type, device_id_, stream);
    }
    template<typename T>
    auto get_output(Batch& batch, std::string const& name, std::optional<MemoryType> const& mem_type) const {
      return get_output<T>(name, mem_type, device_id_, default_stream_);
    }
    template<typename T>
    auto get_output(Batch& batch, std::string const& name) const {
      return get_output<T>(name, preferred_mem_type(batch), device_id_, default_stream_);
    }

    /**
     * @brief Retrieve value of configuration parameter
     */
    template <typename T>
    auto get_config_param(std::string const& name) const {
      return shared_state_->get_config_param<T>(name);
    }
    template <typename T>
    auto get_config_param(std::string const& name, T default_value) const {
      return shared_state_->get_config_param<T>(name, default_value);
    }

    Model(std::shared_ptr<SharedState> shared_state, device_id_t device_id, cudaStream_t default_stream, DeploymentType deployment_type, std::string const& filepath) :
      shared_state_{shared_state}, device_id_{device_id}, default_stream_{default_stream}, deployment_type_{deployment_type} filepath_{filepath} {}

    auto get_device_id() const { return device_id_; }
    auto get_deployment_type() const { return deployment_type_; }
    auto const& get_filepath() const { return filepath_; }

    protected:
      auto get_shared_state() const { return shared_state_; }

    private:
      std::shared_ptr<SharedState> shared_state_;
      device_id_t device_id_;
      cudaStream_t default_stream_;
      DeploymentType deployment_type_;
      std::string filepath_;
  };
}}}  // namespace triton::backend::rapids
