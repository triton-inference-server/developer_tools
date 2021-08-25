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
#include <string>
#include <rapids_triton/memory/types.hpp>
#include <rapids_triton/triton/device.hpp>
#include <rapids_triton/triton/requests.hpp>
#include <triton/backend/backend_input_collector.h>
#include <triton/backend/backend_output_responder.h>

namespace triton { namespace backend { namespace rapids {
  struct Batch {
    using size_type = std::size_t;

    Batch(TRITONBACKEND_Request** raw_requests, request_size_t count, TRITONBACKEND_MemoryManager& triton_mem_manager, bool use_pinned_input, bool use_pinned_output, size_type max_batch_size, cudaStream_t stream) :
      requests_(raw_requests, count),
      responses_(construct_responses(requests_.begin(), requests_.end())),
      collector_{
        raw_requests,
        count,
        &responses_,
        &triton_mem_manager,
        use_pinned_input,
        stream
      },
      responder_{
        raw_requests,
        count,
        &responses_,
        max_batch_size,
        use_pinned_output,
        stream
      } {}


    template<typename T>
    auto get_input(std::string const& name, MemoryType memory_type, device_id_t device_id, cudaStream_t stream) {
      collector_.ProcessTensor(
        name.c_str(),
        nullptr // Return data without copy if possible
        // TODO
      );

    }

    template<typename T>
    auto get_output(std::string name, MemoryType memory_type, device_id_t device_id, cudaStream_t stream) {
      // TODO(wphicks)
    }

    private:
      std::vector<TRITONBACKEND_Request*> requests_;
      std::vector<TRITONBACKEND_Response*> responses_;
      BackendInputCollector collector_;
      BackendOutputResponder responder_;

      auto get_shape() {
      }
  };
}}}  // namespace triton::backend::rapids

