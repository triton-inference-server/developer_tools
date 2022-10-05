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
#include <algorithm>
#include <iterator>
#include <triton/developer_tools/exceptions.hpp>
#include <triton/developer_tools/triton/logging.hpp>
#include <vector>

namespace triton {
namespace developer_tools {
namespace backend {

template <typename Iter>
auto construct_responses(Iter requests_begin, Iter requests_end)
{
  auto responses = std::vector<TRITONBACKEND_Response*>{};

  auto requests_size = std::distance(requests_begin, requests_end);
  if (!(requests_size > 0)) {
    throw TritonException(Error::Internal,
                          "Invalid iterators for requests when constructing responses");
  }
  responses.reserve(requests_size);

  std::transform(requests_begin, requests_end, std::back_inserter(responses), [](auto* request) {
    auto* response = static_cast<TRITONBACKEND_Response*>(nullptr);
    triton_check(TRITONBACKEND_ResponseNew(&response, request));
    return response;
  });
  return responses;
}

template <typename Iter>
void send_responses(Iter begin, Iter end, TRITONSERVER_Error* err)
{
  std::for_each(begin, end, [err](auto& response) {
    decltype(err) err_copy;
    if (err != nullptr) {
      err_copy = TRITONSERVER_ErrorNew(TRITONSERVER_ErrorCode(err), TRITONSERVER_ErrorMessage(err));
    } else {
      err_copy = err;
    }

    if (response == nullptr) {
      log_error(__FILE__, __LINE__) << "Failure in response collation";
    } else {
      try {
        triton_check(
          TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err_copy));
      } catch (TritonException& err) {
        log_error(__FILE__, __LINE__, err.what());
      }
    }
  });
}

}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
