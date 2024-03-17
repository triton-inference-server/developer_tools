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
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <triton/developer_tools/batch/batch.hpp>
#include <triton/developer_tools/exceptions.hpp>
#include <triton/developer_tools/triton/model.hpp>
#include <triton/developer_tools/triton/model_instance.hpp>
#include <triton/developer_tools/triton/statistics.hpp>
#include <triton/developer_tools/utils/narrow.hpp>
#include <vector>

namespace triton {
namespace developer_tools {
namespace backend {
namespace triton_api {
template <typename ModelState, typename ModelInstanceState>
auto* execute(TRITONBACKEND_ModelInstance* instance,
              TRITONBACKEND_Request** raw_requests,
              std::size_t request_count)
{
  auto start_time = std::chrono::steady_clock::now();

  auto* result = static_cast<TRITONSERVER_Error*>(nullptr);

  try {
    auto* model_state    = get_model_state<ModelState>(*get_model_from_instance(*instance));
    auto* instance_state = get_instance_state<ModelInstanceState>(*instance);
    auto& model          = instance_state->get_model();
    auto max_batch_size  = model.template get_config_param<std::size_t>("max_batch_size");

    /* Note: It is safe to keep a reference to the model in this closure
     * and a pointer to the instance in the next because the batch goes
     * out of scope at the end of this block and Triton guarantees that
     * the liftimes of both the instance and model extend beyond this
     * function call. */
    auto output_shape_fetcher = [&model](std::string const& name, Batch::size_type batch_dim) {
      auto result       = std::vector<Batch::size_type>{};
      auto config_shape = model.get_output_shape(name);
      if (config_shape.size() > 0 && config_shape[0] < 0) { config_shape[0] = batch_dim; }
      std::transform(
        std::begin(config_shape),
        std::end(config_shape),
        std::back_inserter(result),
        [](auto& coord) {
          if (coord < 0) {
            throw TritonException(
              Error::Internal,
              "Backends with variable-shape outputs must request desired output shape");
          } else {
            return narrow<std::size_t>(coord);
          }
        });
      return result;
    };
    auto statistics_reporter = [instance](TRITONBACKEND_Request* request,
                                          time_point req_start,
                                          time_point req_comp_start,
                                          time_point req_comp_end,
                                          time_point req_end) {
      report_statistics(*instance, *request, req_start, req_comp_start, req_comp_end, req_end);
    };

    auto batch = Batch(raw_requests,
                       request_count,
                       *(model_state->TritonMemoryManager()),
                       std::move(output_shape_fetcher),
                       std::move(statistics_reporter),
                       model_state->EnablePinnedInput(),
                       model_state->EnablePinnedOutput(),
                       max_batch_size,
                       model.get_stream());

    if constexpr (IS_GPU_BUILD) {
      if (model.get_deployment_type() == GPUDeployment) {
        cuda_check(cudaSetDevice(model.get_device_id()));
      }
    }

    auto predict_err = static_cast<TRITONSERVER_Error*>(nullptr);
    try {
      model.predict(batch);
    } catch (TritonException& err) {
      predict_err = err.error();
    }

    auto& compute_start_time = batch.compute_start_time();
    auto compute_end_time    = std::chrono::steady_clock::now();
    batch.finalize(predict_err);
    auto end_time = std::chrono::steady_clock::now();

    report_statistics(
      *instance, request_count, start_time, compute_start_time, compute_end_time, end_time);
  } catch (TritonException& err) {
    result = err.error();
  }

  return result;
}
}  // namespace triton_api
}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
