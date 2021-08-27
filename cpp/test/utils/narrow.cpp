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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <rapids_triton/utils/narrow.hpp>
#include <string>

namespace triton {
namespace backend {
namespace rapids {
TEST(RapidsTriton, narrow) {
  EXPECT_THROW(narrow<std::size_t>(-1), TritonException);
  narrow<std::size_t>(int{5});
  EXPECT_THROW(narrow<int>(std::numeric_limits<std::size_t>::max()),
               TritonException);
  narrow<int>(std::size_t{5});
}

}  // namespace rapids
}  // namespace backend
}  // namespace triton
