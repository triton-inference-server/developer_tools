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

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <algorithm>
#include <chrono>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <rapids_triton/memory/buffer.hpp>
#include <rapids_triton/memory/types.hpp>
#include <rapids_triton/tensor/tensor.hpp>
#include <rapids_triton/triton/device.hpp>
#include <rapids_triton/triton/input.hpp>
#include <rapids_triton/triton/requests.hpp>
#include <rapids_triton/triton/responses.hpp>
#include <rapids_triton/triton/statistics.hpp>
#include <rapids_triton/utils/narrow.hpp>
#include <triton/backend/backend_input_collector.h>
#include <triton/backend/backend_output_responder.h>

namespace triton { namespace backend { namespace rapids {
  /**
   * @brief A representation of all data about a single batch of inference
   * requests
   *
   * Batch objects are the primary interface point between rapids_triton Models
   * and the Triton server itself. By calling the `get_input` and `get_output`
   * methods of a batch, Model implementations can retrieve the input Tensors
   * necessary for prediction and the output Tensors where results can be
   * stored.
   *
   * Batch objects also handle a variety of other tasks necessary for
   * processing a batch in the Triton model. This includes reporting statistics
   * on how long it took to process requests and sending responses to the
   * client via the Triton server once processing is complete.
   *
   * It is not recommended that developers of rapids_triton backends try to
   * construct Batch objects directly. Instead, you should make use of the
   * rapids::triton_api::execute template, which will construct the Batch for
   * you.
   */
  struct Batch {
    using size_type = std::size_t;

    Batch(TRITONBACKEND_Request** raw_requests,
          request_size_t count,
          TRITONBACKEND_MemoryManager& triton_mem_manager,
          std::function<std::vector<size_type>(std::string const&, size_type)> get_output_shape,
          std::function<
            void(
              TRITONBACKEND_Request*,
              time_point const&,
              time_point const&,
              time_point const&,
              time_point const&)> report_request_statistics,
          bool use_pinned_input,
          bool use_pinned_output,
          size_type max_batch_size,
          cudaStream_t stream) :
    requests_(raw_requests, raw_requests + count),
    responses_(construct_responses(requests_.begin(), requests_.end())),
    get_output_shape_{get_output_shape},
    report_statistics_{report_request_statistics},
    collector_(
      raw_requests,
      count,
      &responses_,
      &triton_mem_manager,
      use_pinned_input,
      stream
    ),
    responder_{std::make_shared<BackendOutputResponder>(
      raw_requests,
      count,
      &responses_,
      max_batch_size,
      &triton_mem_manager,
      use_pinned_output,
      stream
    )},
    stream_{stream},
    start_time_{std::chrono::steady_clock::now()},
    compute_start_time_{std::chrono::steady_clock::now()},
    batch_size_{} {}

    template<typename T>
    auto get_input_shape(std::string const& name) {
      auto result = std::vector<size_type>{};
      if(!requests_.empty()) {
        result = get_triton_input_shape<T>(std::begin(requests_), std::end(requests_), name);

        auto input_batch_dim = size_type{};
        if (result.size() > 0) {
          input_batch_dim = result[0];
        } else {
          input_batch_dim = size_type{};
        }

        if(batch_size_.has_value()) {
          if(batch_size_.value() != input_batch_dim) {
            throw TritonException(
                Error::Internal, "all input tensors must have same batch dimension");
          }
        } else {
          batch_size_ = input_batch_dim;
        }
      }
      return result;
    }


    template<typename T>
    auto get_input(std::string const& name, std::optional<MemoryType>
        memory_type, device_id_t device_id, cudaStream_t stream) {
      auto shape = get_input_shape<T>(name);
      auto size_bytes = sizeof(T) * std::reduce(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
      auto allowed_memory_configs = std::vector<std::pair<MemoryType, int64_t>>{};
      if (memory_type.has_value()) {
        allowed_memory_configs.emplace_back(memory_type.value(), device_id);
      } else {
        allowed_memory_configs.emplace_back(HostMemory, int64_t{});
        allowed_memory_configs.emplace_back(DeviceMemory, device_id);
      }

      auto const* raw_buffer = static_cast<char*>(nullptr);
      auto reported_bytes = std::size_t{};
      auto reported_mem_type = MemoryType{};
      auto reported_device_id = int64_t{};

      triton_check(collector_.ProcessTensor(
        name.c_str(),
        static_cast<char*>(nullptr), // Return data without copy if possible
        size_bytes,
        allowed_memory_configs,
        &raw_buffer,
        &reported_bytes,
        &reported_mem_type,
        &reported_device_id
      ));

      auto buffer = Buffer(
        reinterpret_cast<T*>(raw_buffer),
        reported_bytes / sizeof(T),
        reported_mem_type,
        reported_device_id,
        stream
      );

      if (memory_type && (reported_mem_type != memory_type || reported_device_id != device_id)) {
        throw TritonException(Error::Internal, "data collected in wrong location");
      }

      // Set start time of batch to time latest input tensor was retrieved
      compute_start_time_ = std::chrono::steady_clock::now();

      return Tensor(std::move(shape), std::move(buffer));
    }

    template<typename T>
    auto get_input(std::string const& name, std::optional<MemoryType>
        memory_type, device_id_t device_id) {
      return get_input<T>(name, memory_type, device_id, stream_);
    }

    template<typename T>
    auto get_output(std::string const& name, std::optional<MemoryType> memory_type, device_id_t device_id, cudaStream_t stream) {
      if (!batch_size_.has_value()) {
        throw TritonException(
          Error::Internal,
          "At least one input must be retrieved before any output"
        );
      }
      auto shape = get_output_shape_(name, batch_size_.value());
      auto buffer_size = std::reduce(
          shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
      auto final_memory_type = MemoryType{};
      if (memory_type.has_value()) {
        final_memory_type = memory_type.value();
      } else {
        // If consumer doesn't care, use HostMemory to avoid additional copy on
        // non-shared-memory responses.
        final_memory_type = HostMemory;
      }
      auto buffer = Buffer<T>(buffer_size, final_memory_type, device_id, stream);
      return OutputTensor<T>(std::move(shape), std::move(buffer), name, responder_);
    }

    template<typename T>
    auto get_output(std::string const& name, std::optional<MemoryType> memory_type, device_id_t device_id) {
      return get_output<T>(name, memory_type, device_id, stream_);
    }

    auto const& compute_start_time() const {
      return compute_start_time_;
    }

    auto stream() const {
      return stream_;
    }

    void finalize(TRITONSERVER_Error* err) {
      auto compute_end_time = std::chrono::steady_clock::now();
      if (responder_->Finalize()) {
        cuda_check(cudaStreamSynchronize(stream_));
      }

      send_responses(std::begin(responses_), std::end(responses_), err);

      // Triton resumes ownership of failed requests; only release on success
      if (err == nullptr) {
        std::for_each(
          std::begin(requests_),
          std::end(requests_),
          [this, &compute_end_time](
              auto& request) {
            report_statistics_(
              request,
              start_time_,
              compute_start_time_,
              compute_end_time,
              std::chrono::steady_clock::now()
            );
          }
        );
        release_requests(std::begin(requests_), std::end(requests_));
      }
    }

    private:
      std::vector<TRITONBACKEND_Request*> requests_;
      std::vector<TRITONBACKEND_Response*> responses_;
      std::function<std::vector<size_type>(std::string const&, size_type)> get_output_shape_;
      std::function<void(TRITONBACKEND_Request*, time_point const&, time_point const&, time_point const&, time_point const&)> report_statistics_;
      BackendInputCollector collector_;
      std::shared_ptr<BackendOutputResponder> responder_;
      cudaStream_t stream_;
      std::chrono::time_point<std::chrono::steady_clock> start_time_;
      std::chrono::time_point<std::chrono::steady_clock> compute_start_time_;
      std::optional<size_type> batch_size_;
  };
}}}  // namespace triton::backend::rapids

