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
#include <algorithm>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/tensor/dtype.hpp>
#include <rapids_triton/utils/narrow.hpp>
#include <triton/core/tritonbackend.h>

namespace triton { namespace backend { namespace rapids {
  inline auto* get_triton_input(TRITONBACKEND_Request* request, std::string const& name) {
    auto result = static_cast<TRITONBACKEND_Input*>(nullptr);
    triton_check(
      TRITONBACKEND_RequestInput(request, name.c_str(), &result));
    return result;
  }

  template<typename T, typename Iter>
  auto get_triton_input_shape(Iter requests_begin, Iter requests_end, std::string const& name) {
    auto result = std::vector<std::size_t>{};

    auto reported_dtype = DType{};
    auto const* input_shape = static_cast<int64_t*>(nullptr);
    auto input_dims = uint32_t{};

    auto batch_dim = std::accumulate(
      requests_begin,
      requests_end,
      int64_t{},
      [&reported_dtype, &input_shape, &input_dims, &name](auto total, auto& request) {
        auto* input = get_triton_input(request, name);
        triton_check(
          TRITONBACKEND_InputProperties(
            input, nullptr, &reported_dtype, &input_shape,
            &input_dims, nullptr, nullptr));

        if (reported_dtype != TritonDtype<T>::value) {
          auto log_stream = std::stringstream{};
          log_stream << "incorrect type "
                     << reported_dtype
                     << " for input with required type "
                     << TritonDtype<T>::value;
          throw(TritonException(Error::Internal, log_stream.str()));
        }

        if (input_dims != 0) {
          total += *input_shape;
        }
        return total;
    });

    result.reserve(input_dims);
    std::transform(
      input_shape,
      input_shape + input_dims,
      std::back_inserter(result),
      [](auto& val) { return narrow<std::size_t>(val); }
    );

    if (!result.empty()) {
      result[0] = narrow<std::size_t>(batch_dim);
    }

    return result;
  }
}}}
