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
#include <triton/backend/backend_common.h>

namespace triton { namespace backend { namespace rapids { namespace triton_api {
  template<typename ModelState, typename ModelInstanceState>
  auto* execute(TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** raw_requests, std::size_t request_count) {
    auto start_time = std::chrono::steady_clock::now();

    auto* result = static_cast<TRITONSERVER_Error*>(nullptr);

    try {
      auto* model_state = rapids::get_model_state<ModelState>(*instance);
      auto* instance_state =
          rapids::get_instance_state<ModelInstanceState>(*instance);
      auto& model = instance_state->get_model();
      auto max_batch_size = model.get_config_param("max_batch_size");
      auto batch = Batch{
          raw_requests, request_count, model_state->TritonMemoryManager(),
          /* Note: It is safe to keep a copy of the model_state
           * pointer in this closure and the instance pointer in the next because
           * the batch goes out of scope at the end of this block and Triton
           * guarantees that the liftimes of both the instance and model states
           * extend beyond this function call. */
          [model_state](std::string const& name) {
            auto result = std::vector<size_type>{};
            auto& triton_result =
                model_state->FindBatchOutput(name)->OutputShape();
            std::transform(std::begin(triton_result), std::end(triton_result),
                           std::back_inserter(result), [](auto& coord) {
                             return narrow<std::size_t>(coord);
                           });
            return result;
          },
          [instance](TRITONBACKEND_Request* request, time_point req_start,
                     time_point req_comp_start, time_point req_comp_end,
                     time_point req_end) {
            report_statistics(*instance, request, req_start, req_comp_start,
                              req_comp_end, req_end);
          },
          model_state->EnablePinnedInput(), model_state->EnablePinnedOutput(),
          max_batch_size model.get_stream()};

      model.predict(batch);
      auto& compute_start_time = batch.compute_start_time();
      auto compute_end_time = std::chrono::steady_clock::now();
      batch.finalize();
      auto end_time = std::chrono::steady_clock::now();

      report_statistics(*instance, request_count, start_time, compute_start_time,
                        compute_end_time, end_time);
    } catch (rapids::TritonException& err) {
      result = err.error();
    }

    return result;
  }
}}}}
