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

#include <gtest/gtest.h>

#include <triton/developer_tools/utils/const_agnostic.hpp>
#include <type_traits>

namespace triton {
namespace developer_tools {
namespace backend {
TEST(BackendTools, const_agnostic)
{
  static_assert(std::is_same<const_agnostic_same_t<bool const, bool>, void>::value);
  static_assert(std::is_same<const_agnostic_same_t<bool, bool>, void>::value);
}

}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
