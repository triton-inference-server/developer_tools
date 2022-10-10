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

#include <triton/core/tritonbackend.h>
#include <chrono>
#include <cstddef>
#include <triton_backend/exceptions.hpp>

namespace triton {
namespace backend {
namespace rapids {
using time_point = std::chrono::time_point<std::chrono::steady_clock>;

/**
 * @brief Report inference statistics for a single request
 *
 * @param instance The Triton model instance which is processing this request
 * @param request The Triton request object itself
 * @param start_time The time at which the backend first received the request
 * @param compute_start_time The time at which the backend began actual
 * inference on the request
 * @param compute_end_time The time at which the backend completed inference
 * on the request
 * @param end_time The time at which the backend finished all processing on
 * the request, including copying out results and returning a response
 */
inline void report_statistics(TRITONBACKEND_ModelInstance& instance,
                              TRITONBACKEND_Request& request,
                              time_point start_time,
                              time_point compute_start_time,
                              time_point compute_end_time,
                              time_point end_time)
{
  triton_check(
    TRITONBACKEND_ModelInstanceReportStatistics(&instance,
                                                &request,
                                                true,
                                                start_time.time_since_epoch().count(),
                                                compute_start_time.time_since_epoch().count(),
                                                compute_end_time.time_since_epoch().count(),
                                                end_time.time_since_epoch().count()));
}

/**
 * @brief Report inference statistics for a batch of requests of given size
 *
 * @param instance The Triton model instance which is processing this batch
 * @param request_count The number of requests in this batch
 * @param start_time The time at which the backend first received the batch
 * @param compute_start_time The time at which the backend began actual
 * inference on the batch
 * @param compute_end_time The time at which the backend completed inference
 * on the batch
 * @param end_time The time at which the backend finished all processing on
 * the batch, including copying out results and returning a response
 */
inline void report_statistics(TRITONBACKEND_ModelInstance& instance,
                              std::size_t request_count,
                              time_point start_time,
                              time_point compute_start_time,
                              time_point compute_end_time,
                              time_point end_time)
{
  triton_check(
    TRITONBACKEND_ModelInstanceReportBatchStatistics(&instance,
                                                     request_count,
                                                     start_time.time_since_epoch().count(),
                                                     compute_start_time.time_since_epoch().count(),
                                                     compute_end_time.time_since_epoch().count(),
                                                     end_time.time_since_epoch().count()));
}
}  // namespace rapids
}  // namespace backend
}  // namespace triton
