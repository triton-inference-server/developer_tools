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
#include <stdint.h>
#include <triton/backend/backend_common.h>

#include <algorithm>
#include <triton/developer_tools/exceptions.hpp>
#include <triton/developer_tools/triton/logging.hpp>

namespace triton {
namespace developer_tools {
namespace backend {
using request_size_t = uint32_t;

template <typename Iter>
void release_requests(Iter begin, Iter end)
{
  std::for_each(begin, end, [](auto& request) {
    try {
      triton_check(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL));
    } catch (TritonException& err) {
      log_error(__FILE__, __LINE__, err.what());
    }
  });
}
}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
