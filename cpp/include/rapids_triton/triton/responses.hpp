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
#include <algorithm>
#include <iterator>
#include <vector>
#include <rapids_triton/exceptions.hpp>
#include <triton/core/tritonbackend.h>

namespace triton { namespace backend { namespace rapids {

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

  std::transform(
      requests_begin, requests_end, std::back_inserter(responses),
      [](auto* request) {
        auto* response = static_cast<TRITONBACKEND_Response*>(nullptr);
        triton_check(TRITONBACKEND_ResponseNew(&response, request));
        return response;
      });
  return responses;
}

}}}  // namespace triton::backend::rapids


