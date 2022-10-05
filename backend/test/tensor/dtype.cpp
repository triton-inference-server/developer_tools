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

#include <triton/developer_tools/tensor/dtype.hpp>

namespace triton {
namespace developer_tools {
namespace backend {

template <DType D>
void check_dtype_conversion()
{
  EXPECT_EQ(D, TritonDtype<typename TritonType<D>::type>::value);
  EXPECT_EQ(D, TritonDtype<typename TritonType<D>::type const>::value);
}

TEST(BackendTools, dtype)
{
  check_dtype_conversion<DTypeBool>();
  check_dtype_conversion<DTypeUint8>();
  check_dtype_conversion<DTypeChar>();
  check_dtype_conversion<DTypeByte>();
  check_dtype_conversion<DTypeUint16>();
  check_dtype_conversion<DTypeUint32>();
  check_dtype_conversion<DTypeUint64>();
  check_dtype_conversion<DTypeInt8>();
  check_dtype_conversion<DTypeInt16>();
  check_dtype_conversion<DTypeInt32>();
  check_dtype_conversion<DTypeInt64>();
  check_dtype_conversion<DTypeFloat32>();
  check_dtype_conversion<DTypeFloat64>();
}

}  // namespace backend
}  // namespace developer_tools
}  // namespace triton
