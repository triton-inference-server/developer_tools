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
#include <cstddef>

#include <rapids_triton/exceptions.hpp>
#include <rapids_triton/utils/narrow.hpp>
#include <triton/common/triton_json.h>

namespace triton { namespace backend { namespace rapids {
  inline auto get_max_batch_size(common::TritonJSON::Value& config) {
    auto reported = int64_t{};
    triton_check(config.MemberAsInt("max_batch_size", &reported));
    return narrow<std::size_t>(reported);
  }
}}}  // namespace triton::backend::rapids
