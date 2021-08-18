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
    Batch(TRITONBACKEND_Request** raw_requests, request_size_t count, ModelState const& model_state, cudaStream_t stream) :
      requests_(raw_requests, count),
      responses_(construct_responses(requests_.begin(), requests_.end())),
      collector_{
        raw_requests,
        count,
        &responses_,
        model_state.TritonMemoryManager(),
        model_state.EnablePinnedInput(),
        stream
      },
      responder_{
        raw_requests,
        count,
        &responses_,
        model_state.MaxBatchSize(),
        model_state.EnablePinnedInput(),
        stream
      } {}


    template<typename T>
    auto get_input(std::string const& name, MemoryType memory_type, device_id_t device_id, cudaStream_t stream) {
      // TODO(wphicks)
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
  }
}}}  // namespace triton::backend::rapids

