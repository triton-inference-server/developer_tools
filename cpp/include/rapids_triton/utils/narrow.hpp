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
#include <rapids_triton/exceptions.hpp>
#include <type_traits>

namespace triton {
namespace backend {
namespace rapids {

template <typename T, typename F>
auto narrow(F from)
{
  auto to = static_cast<T>(from);

  if (static_cast<F>(to) != from ||
      (std::is_signed<F>::value && !std::is_signed<T>::value && from < F{}) ||
      (std::is_signed<T>::value && !std::is_signed<F>::value && to < T{}) ||
      (std::is_signed<T>::value == std::is_signed<F>::value && ((to < T{}) != (from < F{})))) {
    throw TritonException(Error::Internal, "invalid narrowing");
  }
  return to;
}

}  // namespace rapids
}  // namespace backend
}  // namespace triton
